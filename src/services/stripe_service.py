"""Stripe payment processing service.

Production-ready implementation that:
- Uses pre-created Stripe Price IDs (no dynamic product/price creation)
- Handles invoice.paid and invoice.payment_failed for subscription renewals
- Properly manages subscription lifecycle and user access
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.config import get_settings
from models.subscription import Subscription, SubscriptionPlan, PaymentMethod
from models import User

logger = logging.getLogger(__name__)


# Plan to Stripe Price ID mapping
# These must be configured in environment variables after creating prices in Stripe Dashboard
PLAN_PRICE_MAP = {
    "business_starter": "STRIPE_PRICE_BUSINESS_STARTER",
    "business_pro": "STRIPE_PRICE_BUSINESS_PRO",
    "business_scale": "STRIPE_PRICE_BUSINESS_SCALE",
    "enterprise": "STRIPE_PRICE_ENTERPRISE",
}


class StripeService:
    """Service for handling Stripe payment operations.

    This service uses pre-created Stripe Price IDs to avoid creating
    duplicate products/prices on every checkout. Create your products
    and prices in the Stripe Dashboard, then configure the price IDs
    in your environment variables.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.settings = get_settings()
        self._stripe = None

    def _get_stripe(self):
        """Get Stripe client (lazy initialization)."""
        if not self._stripe:
            try:
                import stripe
                stripe.api_key = self.settings.STRIPE_SECRET_KEY
                self._stripe = stripe
            except ImportError:
                raise RuntimeError("stripe package not installed. Run: pip install stripe")
        return self._stripe

    def _get_price_id(self, plan: str) -> str:
        """Get Stripe Price ID for a plan from configuration."""
        env_var = PLAN_PRICE_MAP.get(plan)
        if not env_var:
            raise ValueError(f"Unknown plan: {plan}")

        price_id = getattr(self.settings, env_var, "")
        if not price_id:
            raise ValueError(
                f"Stripe Price ID not configured for plan '{plan}'. "
                f"Set {env_var} in your environment variables."
            )
        return price_id

    async def create_checkout_session(
        self,
        user_id: str,
        plan: str,
        billing_period: str = "monthly",
        trial_days: int = 0
    ) -> Dict[str, Any]:
        """Create a Stripe checkout session for subscription upgrade.

        Args:
            user_id: The user's ID
            plan: Plan name (business_starter, business_pro, business_scale, enterprise)
            billing_period: "monthly" or "yearly" (yearly prices should be separate Price IDs)
            trial_days: Number of trial days (0 = no trial). When > 0, payment method
                collection is deferred until trial end.

        Returns:
            Dict with checkout_url and session_id
        """
        stripe = self._get_stripe()

        # Get user
        user = await self.db.get(User, user_id)
        if not user:
            raise ValueError("User not found")

        # Get the pre-created Price ID from configuration
        price_id = self._get_price_id(plan)

        # Check if user already has a Stripe customer ID
        existing_sub = await self.db.execute(
            select(Subscription).where(Subscription.user_id == user_id)
        )
        existing_subscription = existing_sub.scalar_one_or_none()

        # Build checkout session parameters
        checkout_params = {
            "mode": "subscription",
            "line_items": [{
                "price": price_id,
                "quantity": 1
            }],
            "success_url": f"{self.settings.FRONTEND_URL}/subscription/success?session_id={{CHECKOUT_SESSION_ID}}",
            "cancel_url": f"{self.settings.FRONTEND_URL}/subscription/cancel",
            "metadata": {
                "user_id": user_id,
                "plan": plan,
                "tenant_id": getattr(user, 'tenant_id', None) or ""
            },
            "subscription_data": {
                "metadata": {
                    "user_id": user_id,
                    "plan": plan
                }
            }
        }

        # Configure trial period
        if trial_days > 0:
            checkout_params["subscription_data"]["trial_period_days"] = trial_days
            checkout_params["payment_method_collection"] = "if_required"

        # Use existing customer if available, otherwise let Stripe create one
        if existing_subscription and existing_subscription.stripe_customer_id:
            checkout_params["customer"] = existing_subscription.stripe_customer_id
        else:
            checkout_params["customer_email"] = user.email

        # Create checkout session
        try:
            session = stripe.checkout.Session.create(**checkout_params)

            logger.info(f"Created checkout session {session.id} for user {user_id}, plan {plan}")

            return {
                "checkout_url": session.url,
                "session_id": session.id
            }
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating checkout session: {e}")
            raise

    async def handle_webhook(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Handle Stripe webhook events.

        Supported events:
        - checkout.session.completed: Initial subscription created
        - customer.subscription.updated: Plan changes, status changes
        - customer.subscription.deleted: Subscription canceled
        - invoice.paid: Successful payment (including renewals)
        - invoice.payment_failed: Failed payment attempt
        """
        handlers = {
            "checkout.session.completed": self._handle_checkout_completed,
            "customer.subscription.created": self._handle_subscription_created,
            "customer.subscription.updated": self._handle_subscription_updated,
            "customer.subscription.deleted": self._handle_subscription_deleted,
            "customer.subscription.trial_will_end": self._handle_trial_will_end,
            "invoice.paid": self._handle_invoice_paid,
            "invoice.payment_failed": self._handle_invoice_payment_failed,
        }

        handler = handlers.get(event_type)
        if handler:
            await handler(event_data)
        else:
            logger.debug(f"Unhandled webhook event type: {event_type}")

    async def _handle_checkout_completed(self, session_data: Dict[str, Any]) -> None:
        """Handle successful checkout completion.

        This fires when a customer completes the Checkout Session and
        the subscription is created. We create/update our local subscription
        record and upgrade the user's edition.
        """
        metadata = session_data.get("metadata", {})
        user_id = metadata.get("user_id")
        plan = metadata.get("plan")
        tenant_id = metadata.get("tenant_id")

        if not user_id or not plan:
            logger.error("Missing user_id or plan in checkout session metadata")
            return

        stripe_subscription_id = session_data.get("subscription")
        stripe_customer_id = session_data.get("customer")

        # Create or update subscription record
        result = await self.db.execute(
            select(Subscription).where(Subscription.user_id == user_id)
        )
        subscription = result.scalar_one_or_none()

        # Determine subscription status from Stripe (trialing vs active)
        stripe_sub_status = "active"
        trial_end_ts = None
        if stripe_subscription_id:
            try:
                stripe = self._get_stripe()
                stripe_sub = stripe.Subscription.retrieve(stripe_subscription_id)
                stripe_sub_status = stripe_sub.get("status", "active")
                if stripe_sub.get("trial_end"):
                    trial_end_ts = datetime.fromtimestamp(stripe_sub["trial_end"])
            except Exception:
                logger.warning(f"Could not fetch subscription {stripe_subscription_id} details")

        if not subscription:
            subscription = Subscription(
                id=f"sub_{user_id}",
                user_id=user_id,
                tenant_id=tenant_id if tenant_id else None,
                stripe_subscription_id=stripe_subscription_id,
                stripe_customer_id=stripe_customer_id,
                plan_id=f"plan_{plan}",
                status=stripe_sub_status,
                current_period_start=datetime.utcnow(),
                current_period_end=datetime.utcnow() + timedelta(days=30),
                trial_end=trial_end_ts
            )
            self.db.add(subscription)
        else:
            subscription.stripe_subscription_id = stripe_subscription_id
            subscription.stripe_customer_id = stripe_customer_id
            subscription.plan_id = f"plan_{plan}"
            subscription.status = stripe_sub_status
            subscription.trial_end = trial_end_ts
            subscription.payment_failed_at = None  # Clear any previous failure

        # Update user edition
        user = await self.db.get(User, user_id)
        if user:
            user.edition = self._get_edition_from_plan(plan)
            logger.info(f"Upgraded user {user_id} to {user.edition} edition")

        await self.db.commit()
        logger.info(f"Checkout completed: user={user_id}, plan={plan}, subscription={stripe_subscription_id}")

    async def _handle_subscription_created(self, subscription_data: Dict[str, Any]) -> None:
        """Handle subscription created event.

        This is redundant with checkout.session.completed for new subscriptions,
        but useful for subscriptions created via API or Stripe Dashboard.
        """
        # Most logic is handled in checkout.session.completed
        # This handler ensures we catch subscriptions created outside checkout flow
        stripe_subscription_id = subscription_data.get("id")
        metadata = subscription_data.get("metadata", {})
        user_id = metadata.get("user_id")

        if not user_id:
            logger.debug(f"Subscription {stripe_subscription_id} created without user_id metadata")
            return

        logger.info(f"Subscription created event for user {user_id}: {stripe_subscription_id}")

    async def _handle_subscription_updated(self, subscription_data: Dict[str, Any]) -> None:
        """Handle subscription updates (plan changes, status changes)."""
        stripe_subscription_id = subscription_data.get("id")

        result = await self.db.execute(
            select(Subscription).where(
                Subscription.stripe_subscription_id == stripe_subscription_id
            )
        )
        subscription = result.scalar_one_or_none()

        if not subscription:
            logger.warning(f"Subscription not found for Stripe ID {stripe_subscription_id}")
            return

        # Update subscription status
        new_status = subscription_data.get("status", "active")
        subscription.status = new_status

        # Update period dates
        current_period_end = subscription_data.get("current_period_end")
        if current_period_end:
            subscription.current_period_end = datetime.fromtimestamp(current_period_end)

        current_period_start = subscription_data.get("current_period_start")
        if current_period_start:
            subscription.current_period_start = datetime.fromtimestamp(current_period_start)

        # Handle status-based access changes
        user = await self.db.get(User, subscription.user_id)
        if user:
            if new_status in ("active", "trialing"):
                # Ensure user has correct edition
                plan = subscription.plan_id.replace("plan_", "")
                user.edition = self._get_edition_from_plan(plan)
            elif new_status in ("past_due", "unpaid"):
                # Keep edition but mark subscription as problematic
                logger.warning(f"Subscription {stripe_subscription_id} is {new_status}")
            elif new_status == "canceled":
                # Downgrade to community
                user.edition = "community"

        await self.db.commit()
        logger.info(f"Subscription updated: {stripe_subscription_id} -> status={new_status}")

    async def _handle_subscription_deleted(self, subscription_data: Dict[str, Any]) -> None:
        """Handle subscription cancellation/deletion."""
        stripe_subscription_id = subscription_data.get("id")

        result = await self.db.execute(
            select(Subscription).where(
                Subscription.stripe_subscription_id == stripe_subscription_id
            )
        )
        subscription = result.scalar_one_or_none()

        if not subscription:
            logger.warning(f"Subscription not found for Stripe ID {stripe_subscription_id}")
            return

        subscription.status = "canceled"
        subscription.canceled_at = datetime.utcnow()

        # Downgrade user to community edition
        user = await self.db.get(User, subscription.user_id)
        if user:
            user.edition = "community"
            logger.info(f"Downgraded user {subscription.user_id} to community edition")

        await self.db.commit()
        logger.info(f"Subscription canceled: {stripe_subscription_id}")

    async def _handle_trial_will_end(self, subscription_data: Dict[str, Any]) -> None:
        """Handle trial ending soon notification.

        Fires 3 days before the trial ends. We log the event so downstream
        systems (email, in-app notifications) can alert the user to add a
        payment method before they lose access.
        """
        stripe_subscription_id = subscription_data.get("id")
        trial_end = subscription_data.get("trial_end")
        metadata = subscription_data.get("metadata", {})
        user_id = metadata.get("user_id")

        result = await self.db.execute(
            select(Subscription).where(
                Subscription.stripe_subscription_id == stripe_subscription_id
            )
        )
        subscription = result.scalar_one_or_none()

        if not subscription:
            logger.warning(f"Trial will end but subscription not found: {stripe_subscription_id}")
            return

        trial_end_dt = datetime.fromtimestamp(trial_end) if trial_end else None
        days_remaining = (trial_end_dt - datetime.utcnow()).days if trial_end_dt else 0

        logger.info(
            f"Trial ending soon: subscription={stripe_subscription_id}, "
            f"user={subscription.user_id}, days_remaining={days_remaining}, "
            f"trial_end={trial_end_dt.isoformat() if trial_end_dt else 'unknown'}"
        )

        # TODO: Integrate with notification service to send email/in-app alert
        # e.g. await notification_service.send_trial_ending_notice(
        #     user_id=subscription.user_id, days_remaining=days_remaining
        # )

    async def _handle_invoice_paid(self, invoice_data: Dict[str, Any]) -> None:
        """Handle successful invoice payment.

        This fires for both initial payments and subscription renewals.
        It confirms the payment went through and updates the subscription period.
        """
        stripe_subscription_id = invoice_data.get("subscription")
        if not stripe_subscription_id:
            # One-time invoice, not subscription-related
            return

        result = await self.db.execute(
            select(Subscription).where(
                Subscription.stripe_subscription_id == stripe_subscription_id
            )
        )
        subscription = result.scalar_one_or_none()

        if not subscription:
            logger.warning(f"Invoice paid but subscription not found: {stripe_subscription_id}")
            return

        # Update subscription status to active (in case it was past_due)
        subscription.status = "active"
        subscription.payment_failed_at = None

        # Update period from invoice lines
        lines = invoice_data.get("lines", {}).get("data", [])
        for line in lines:
            period = line.get("period", {})
            if period.get("end"):
                subscription.current_period_end = datetime.fromtimestamp(period["end"])
            if period.get("start"):
                subscription.current_period_start = datetime.fromtimestamp(period["start"])
            break  # Just use the first line item

        # Ensure user has access
        user = await self.db.get(User, subscription.user_id)
        if user:
            plan = subscription.plan_id.replace("plan_", "")
            user.edition = self._get_edition_from_plan(plan)

        await self.db.commit()

        amount_paid = invoice_data.get("amount_paid", 0) / 100  # Convert from cents
        logger.info(
            f"Invoice paid: subscription={stripe_subscription_id}, "
            f"amount=${amount_paid:.2f}, user={subscription.user_id}"
        )

    async def _handle_invoice_payment_failed(self, invoice_data: Dict[str, Any]) -> None:
        """Handle failed invoice payment.

        This fires when a subscription renewal payment fails. We:
        1. Mark the subscription as past_due
        2. Record the failure timestamp
        3. Optionally restrict access after grace period

        Note: Stripe has its own dunning (retry) logic. This handler
        lets us track failures and take action if needed.
        """
        stripe_subscription_id = invoice_data.get("subscription")
        if not stripe_subscription_id:
            return

        result = await self.db.execute(
            select(Subscription).where(
                Subscription.stripe_subscription_id == stripe_subscription_id
            )
        )
        subscription = result.scalar_one_or_none()

        if not subscription:
            logger.warning(f"Invoice failed but subscription not found: {stripe_subscription_id}")
            return

        # Mark payment as failed
        subscription.status = "past_due"
        if not subscription.payment_failed_at:
            subscription.payment_failed_at = datetime.utcnow()

        # Calculate how long payment has been failing
        days_overdue = (datetime.utcnow() - subscription.payment_failed_at).days

        # Grace period: allow access for 7 days after first failure
        # After that, downgrade to community
        GRACE_PERIOD_DAYS = 7

        user = await self.db.get(User, subscription.user_id)
        if user:
            if days_overdue >= GRACE_PERIOD_DAYS:
                user.edition = "community"
                logger.warning(
                    f"User {subscription.user_id} downgraded after {days_overdue} days "
                    f"of failed payments"
                )
            else:
                logger.warning(
                    f"Payment failed for user {subscription.user_id}, "
                    f"grace period: {GRACE_PERIOD_DAYS - days_overdue} days remaining"
                )

        await self.db.commit()

        attempt_count = invoice_data.get("attempt_count", 1)
        logger.warning(
            f"Invoice payment failed: subscription={stripe_subscription_id}, "
            f"attempt={attempt_count}, user={subscription.user_id}"
        )

    def _get_edition_from_plan(self, plan: str) -> str:
        """Get edition name from plan name."""
        if plan.startswith("business"):
            return "business"
        elif plan == "enterprise":
            return "enterprise"
        else:
            return "community"

    async def cancel_subscription(self, user_id: str, immediate: bool = False) -> Dict[str, Any]:
        """Cancel a user's subscription.

        Args:
            user_id: The user's ID
            immediate: If True, cancel immediately. If False, cancel at period end.

        Returns:
            Dict with cancellation details
        """
        stripe = self._get_stripe()

        result = await self.db.execute(
            select(Subscription).where(Subscription.user_id == user_id)
        )
        subscription = result.scalar_one_or_none()

        if not subscription or not subscription.stripe_subscription_id:
            raise ValueError("No active subscription found")

        try:
            if immediate:
                # Cancel immediately
                stripe.Subscription.delete(subscription.stripe_subscription_id)
            else:
                # Cancel at period end
                stripe.Subscription.modify(
                    subscription.stripe_subscription_id,
                    cancel_at_period_end=True
                )

            logger.info(
                f"Subscription cancellation requested: user={user_id}, "
                f"immediate={immediate}"
            )

            return {
                "status": "canceled" if immediate else "cancel_scheduled",
                "effective_date": datetime.utcnow().isoformat() if immediate
                    else subscription.current_period_end.isoformat()
            }
        except stripe.error.StripeError as e:
            logger.error(f"Failed to cancel subscription: {e}")
            raise

    async def get_billing_portal_url(self, user_id: str) -> str:
        """Get Stripe Customer Portal URL for self-service billing management.

        Args:
            user_id: The user's ID

        Returns:
            URL to the Stripe Customer Portal
        """
        stripe = self._get_stripe()

        result = await self.db.execute(
            select(Subscription).where(Subscription.user_id == user_id)
        )
        subscription = result.scalar_one_or_none()

        if not subscription or not subscription.stripe_customer_id:
            raise ValueError("No billing information found")

        try:
            session = stripe.billing_portal.Session.create(
                customer=subscription.stripe_customer_id,
                return_url=f"{self.settings.FRONTEND_URL}/settings/billing"
            )
            return session.url
        except stripe.error.StripeError as e:
            logger.error(f"Failed to create billing portal session: {e}")
            raise

    async def list_invoices(
        self,
        user_id: str,
        limit: int = 10,
        starting_after: str = None
    ) -> Dict[str, Any]:
        """List invoices for a user.

        Args:
            user_id: The user's ID
            limit: Maximum number of invoices to return (1-100)
            starting_after: Invoice ID to paginate after

        Returns:
            Dict with invoices list and pagination info
        """
        stripe = self._get_stripe()

        result = await self.db.execute(
            select(Subscription).where(Subscription.user_id == user_id)
        )
        subscription = result.scalar_one_or_none()

        if not subscription or not subscription.stripe_customer_id:
            return {"invoices": [], "has_more": False}

        try:
            params = {
                "customer": subscription.stripe_customer_id,
                "limit": min(limit, 100),
            }
            if starting_after:
                params["starting_after"] = starting_after

            invoices = stripe.Invoice.list(**params)

            return {
                "invoices": [
                    {
                        "id": inv.id,
                        "number": inv.number,
                        "status": inv.status,
                        "amount_due": inv.amount_due / 100,  # Convert from cents
                        "amount_paid": inv.amount_paid / 100,
                        "currency": inv.currency.upper(),
                        "created": datetime.fromtimestamp(inv.created).isoformat(),
                        "due_date": datetime.fromtimestamp(inv.due_date).isoformat() if inv.due_date else None,
                        "paid_at": datetime.fromtimestamp(inv.status_transitions.paid_at).isoformat() if inv.status_transitions.paid_at else None,
                        "period_start": datetime.fromtimestamp(inv.period_start).isoformat() if inv.period_start else None,
                        "period_end": datetime.fromtimestamp(inv.period_end).isoformat() if inv.period_end else None,
                        "invoice_pdf": inv.invoice_pdf,
                        "hosted_invoice_url": inv.hosted_invoice_url,
                        "description": inv.description,
                    }
                    for inv in invoices.data
                ],
                "has_more": invoices.has_more,
            }
        except stripe.error.StripeError as e:
            logger.error(f"Failed to list invoices: {e}")
            raise

    async def get_invoice(self, user_id: str, invoice_id: str) -> Dict[str, Any]:
        """Get a single invoice by ID.

        Args:
            user_id: The user's ID (for authorization)
            invoice_id: The Stripe invoice ID

        Returns:
            Invoice details
        """
        stripe = self._get_stripe()

        # Verify the user owns this invoice
        result = await self.db.execute(
            select(Subscription).where(Subscription.user_id == user_id)
        )
        subscription = result.scalar_one_or_none()

        if not subscription or not subscription.stripe_customer_id:
            raise ValueError("No billing information found")

        try:
            inv = stripe.Invoice.retrieve(invoice_id)

            # Security check: ensure invoice belongs to this customer
            if inv.customer != subscription.stripe_customer_id:
                raise ValueError("Invoice not found")

            return {
                "id": inv.id,
                "number": inv.number,
                "status": inv.status,
                "amount_due": inv.amount_due / 100,
                "amount_paid": inv.amount_paid / 100,
                "amount_remaining": inv.amount_remaining / 100,
                "currency": inv.currency.upper(),
                "created": datetime.fromtimestamp(inv.created).isoformat(),
                "due_date": datetime.fromtimestamp(inv.due_date).isoformat() if inv.due_date else None,
                "paid_at": datetime.fromtimestamp(inv.status_transitions.paid_at).isoformat() if inv.status_transitions.paid_at else None,
                "period_start": datetime.fromtimestamp(inv.period_start).isoformat() if inv.period_start else None,
                "period_end": datetime.fromtimestamp(inv.period_end).isoformat() if inv.period_end else None,
                "invoice_pdf": inv.invoice_pdf,
                "hosted_invoice_url": inv.hosted_invoice_url,
                "description": inv.description,
                "subtotal": inv.subtotal / 100,
                "tax": inv.tax / 100 if inv.tax else 0,
                "total": inv.total / 100,
                "line_items": [
                    {
                        "description": item.description,
                        "amount": item.amount / 100,
                        "quantity": item.quantity,
                        "period_start": datetime.fromtimestamp(item.period.start).isoformat() if item.period else None,
                        "period_end": datetime.fromtimestamp(item.period.end).isoformat() if item.period else None,
                    }
                    for item in inv.lines.data
                ],
            }
        except stripe.error.StripeError as e:
            logger.error(f"Failed to retrieve invoice: {e}")
            raise

    async def get_upcoming_invoice(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get the upcoming invoice for a subscription.

        This shows what the user will be charged at the next billing cycle.

        Args:
            user_id: The user's ID

        Returns:
            Upcoming invoice details, or None if no upcoming invoice
        """
        stripe = self._get_stripe()

        result = await self.db.execute(
            select(Subscription).where(Subscription.user_id == user_id)
        )
        subscription = result.scalar_one_or_none()

        if not subscription or not subscription.stripe_customer_id:
            return None

        try:
            inv = stripe.Invoice.upcoming(customer=subscription.stripe_customer_id)

            return {
                "amount_due": inv.amount_due / 100,
                "currency": inv.currency.upper(),
                "next_payment_attempt": datetime.fromtimestamp(inv.next_payment_attempt).isoformat() if inv.next_payment_attempt else None,
                "period_start": datetime.fromtimestamp(inv.period_start).isoformat() if inv.period_start else None,
                "period_end": datetime.fromtimestamp(inv.period_end).isoformat() if inv.period_end else None,
                "subtotal": inv.subtotal / 100,
                "tax": inv.tax / 100 if inv.tax else 0,
                "total": inv.total / 100,
                "line_items": [
                    {
                        "description": item.description,
                        "amount": item.amount / 100,
                        "quantity": item.quantity,
                    }
                    for item in inv.lines.data
                ],
            }
        except stripe.error.InvalidRequestError:
            # No upcoming invoice (e.g., subscription canceled)
            return None
        except stripe.error.StripeError as e:
            logger.error(f"Failed to retrieve upcoming invoice: {e}")
            raise
