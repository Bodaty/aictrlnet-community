"""Stripe payment processing service."""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.config import get_settings
from models.subscription import Subscription, SubscriptionPlan, PaymentMethod
from models import User
from adapters.factory import AdapterFactory
from adapters.models import AdapterConfig, AdapterRequest, AdapterCategory
from schemas.license import LicenseUpgradeRequest

logger = logging.getLogger(__name__)


class StripeService:
    """Service for handling Stripe payment operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.settings = get_settings()
        self._adapter = None
    
    async def _get_adapter(self):
        """Get or create Stripe adapter."""
        if not self._adapter:
            config = AdapterConfig(
                name="stripe",
                category=AdapterCategory.PAYMENT,
                api_key=self.settings.STRIPE_SECRET_KEY,
                timeout_seconds=30
            )
            self._adapter = await AdapterFactory.create_adapter("stripe", config)
            await self._adapter.initialize()
        return self._adapter
    
    async def create_checkout_session(
        self,
        user_id: str,
        plan: str,
        billing_period: str = "monthly"
    ) -> Dict[str, Any]:
        """Create a Stripe checkout session for subscription upgrade."""
        
        # Get user
        user = await self.db.get(User, user_id)
        if not user:
            raise ValueError("User not found")
        
        # Get plan details
        plan_result = await self.db.execute(
            select(SubscriptionPlan).where(SubscriptionPlan.name == plan)
        )
        subscription_plan = plan_result.scalar_one_or_none()
        
        if not subscription_plan:
            # Create default plan if it doesn't exist
            subscription_plan = SubscriptionPlan(
                id=f"plan_{plan}",
                name=plan,
                display_name=plan.replace("_", " ").title(),
                price_monthly=self._get_plan_price(plan, "monthly"),
                price_yearly=self._get_plan_price(plan, "yearly"),
                features={},
                limits={}
            )
            self.db.add(subscription_plan)
            await self.db.commit()
        
        # Calculate price
        price = subscription_plan.price_monthly if billing_period == "monthly" else subscription_plan.price_yearly
        
        # Create checkout session with Stripe
        adapter = await self._get_adapter()
        
        # Create product if needed
        product_request = AdapterRequest(
            operation="create_product",
            parameters={
                "name": subscription_plan.display_name,
                "description": f"AICtrlNet {subscription_plan.display_name} Subscription",
                "metadata": {
                    "plan_id": subscription_plan.id,
                    "edition": plan
                }
            }
        )
        
        product_response = await adapter.execute(product_request)
        product_id = product_response.data.get("id")
        
        # Create price
        price_request = AdapterRequest(
            operation="create_price",
            parameters={
                "product": product_id,
                "unit_amount": int(price * 100),  # Convert to cents
                "currency": "usd",
                "recurring": {
                    "interval": "month" if billing_period == "monthly" else "year"
                }
            }
        )
        
        price_response = await adapter.execute(price_request)
        price_id = price_response.data.get("id")
        
        # Create checkout session
        checkout_request = AdapterRequest(
            operation="create_checkout_session",
            parameters={
                "customer_email": user.email,
                "mode": "subscription",
                "line_items": [{
                    "price": price_id,
                    "quantity": 1
                }],
                "success_url": f"{self.settings.FRONTEND_URL}/license/success?session_id={{CHECKOUT_SESSION_ID}}",
                "cancel_url": f"{self.settings.FRONTEND_URL}/license/upgrade",
                "metadata": {
                    "user_id": user_id,
                    "plan": plan,
                    "tenant_id": user.tenant_id
                }
            }
        )
        
        checkout_response = await adapter.execute(checkout_request)
        
        return {
            "checkout_url": checkout_response.data.get("url"),
            "session_id": checkout_response.data.get("id")
        }
    
    async def handle_webhook(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Handle Stripe webhook events."""
        
        if event_type == "checkout.session.completed":
            await self._handle_checkout_completed(event_data)
        elif event_type == "customer.subscription.updated":
            await self._handle_subscription_updated(event_data)
        elif event_type == "customer.subscription.deleted":
            await self._handle_subscription_deleted(event_data)
    
    async def _handle_checkout_completed(self, session_data: Dict[str, Any]) -> None:
        """Handle successful checkout completion."""
        
        # Extract metadata
        metadata = session_data.get("metadata", {})
        user_id = metadata.get("user_id")
        plan = metadata.get("plan")
        tenant_id = metadata.get("tenant_id")
        
        if not user_id or not plan:
            logger.error("Missing user_id or plan in checkout session metadata")
            return
        
        # Get subscription ID from session
        subscription_id = session_data.get("subscription")
        
        # Create or update subscription record
        subscription = await self.db.get(Subscription, f"sub_{user_id}")
        
        if not subscription:
            subscription = Subscription(
                id=f"sub_{user_id}",
                user_id=user_id,
                tenant_id=tenant_id,
                stripe_subscription_id=subscription_id,
                plan_id=f"plan_{plan}",
                status="active",
                current_period_start=datetime.utcnow(),
                current_period_end=datetime.utcnow() + timedelta(days=30)
            )
            self.db.add(subscription)
        else:
            subscription.stripe_subscription_id = subscription_id
            subscription.plan_id = f"plan_{plan}"
            subscription.status = "active"
        
        # Update user edition
        user = await self.db.get(User, user_id)
        if user:
            user.edition = self._get_edition_from_plan(plan)
        
        await self.db.commit()
        logger.info(f"Subscription created/updated for user {user_id} with plan {plan}")
    
    async def _handle_subscription_updated(self, subscription_data: Dict[str, Any]) -> None:
        """Handle subscription updates."""
        
        stripe_subscription_id = subscription_data.get("id")
        
        # Find subscription by Stripe ID
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
        subscription.status = subscription_data.get("status", "active")
        subscription.current_period_end = datetime.fromtimestamp(
            subscription_data.get("current_period_end", 0)
        )
        
        await self.db.commit()
    
    async def _handle_subscription_deleted(self, subscription_data: Dict[str, Any]) -> None:
        """Handle subscription cancellation."""
        
        stripe_subscription_id = subscription_data.get("id")
        
        # Find subscription by Stripe ID
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
        subscription.status = "canceled"
        subscription.canceled_at = datetime.utcnow()
        
        # Downgrade user to community edition
        user = await self.db.get(User, subscription.user_id)
        if user:
            user.edition = "community"
        
        await self.db.commit()
    
    def _get_plan_price(self, plan: str, period: str) -> Decimal:
        """Get price for a plan."""
        prices = {
            "business_starter": {"monthly": Decimal("49"), "yearly": Decimal("490")},
            "business_growth": {"monthly": Decimal("99"), "yearly": Decimal("990")},
            "business_scale": {"monthly": Decimal("199"), "yearly": Decimal("1990")},
            "enterprise": {"monthly": Decimal("499"), "yearly": Decimal("4990")}
        }
        
        return prices.get(plan, {}).get(period, Decimal("0"))
    
    def _get_edition_from_plan(self, plan: str) -> str:
        """Get edition name from plan name."""
        if plan.startswith("business"):
            return "business"
        elif plan == "enterprise":
            return "enterprise"
        else:
            return "community"