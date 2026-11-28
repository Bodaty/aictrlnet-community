"""How each plan is sold. The one place the code says so.

Source: the "Current pricing" section of
docs/misc/PRICING_AND_LICENSING_STRATEGY_2025.md (decided 25 Nov 2026).

- Business Starter, Growth and Scale are sold by card (Stripe checkout).
- Enterprise Cloud is contact-sales, shown as "From $12,000/mo".
- On-Premise and Government are contact-sales and quoted per deal.
- Community (self-hosted, open source) is free.
- Community Cloud and Team are not sold; their rows stay in the catalog,
  inactive, until plan limits follow each tenant's plan.

Prices for card plans live on the plan rows. Contact-sales plans carry no
price in the API: the frontend shows ``price_display`` instead.
"""

from typing import Optional

FREE = "free"
SELF_SERVE = "self_serve"
CONTACT_SALES = "contact_sales"
NOT_SOLD = "not_sold"

SELF_SERVE_PLANS = ("business_starter", "business_growth", "business_scale")

_CONTACT_SALES_DISPLAY = {
    "enterprise-cloud": "From $12,000/mo",
    "business_onprem": "Quoted",
    "enterprise-onprem": "Quoted",
    "government": "Quoted",
}

NOT_SOLD_PLANS = ("community_cloud", "team", "enterprise_onprem")

# The decided expert hours for each card plan (pricing doc), and the add-on rate.
EXPERT_HOURS = {"business_starter": 2, "business_growth": 5, "business_scale": 8}
EXTRA_EXPERT_HOUR_USD = 295


def sales_model(plan_name: Optional[str]) -> str:
    name = (plan_name or "").strip()
    if name in SELF_SERVE_PLANS:
        return SELF_SERVE
    if name in _CONTACT_SALES_DISPLAY:
        return CONTACT_SALES
    if name in ("community", "community_free"):
        return FREE
    return NOT_SOLD


def price_display(plan_name: Optional[str]) -> Optional[str]:
    """The label a contact-sales plan shows in place of a price."""
    return _CONTACT_SALES_DISPLAY.get((plan_name or "").strip())
