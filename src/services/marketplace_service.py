"""Community Marketplace Service — browse, search, install (limit 10), get item.

Community edition provides read-only marketplace browsing, search, and
installation with a per-user limit of 10 active installations.
Publishing is available in the Business edition.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select, func, or_, and_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession

from models.marketplace import MarketplaceItem, MarketplaceReview, MarketplaceInstallation
from schemas.marketplace import (
    MarketplaceItemCreate,
    MarketplaceItemUpdate,
    MarketplaceItemResponse,
    MarketplaceItemListResponse,
    MarketplaceSearchRequest,
    MarketplaceReviewCreate,
    MarketplaceReviewResponse,
    MarketplaceReviewListResponse,
    MarketplaceInstallRequest,
    MarketplaceInstallResponse,
    MarketplaceInstallationListResponse,
    MarketplaceUninstallResponse,
)

logger = logging.getLogger(__name__)

# Community edition: maximum active installations per user
COMMUNITY_INSTALL_LIMIT = 10


class MarketplaceService:
    """Community marketplace service with browse, search, install, and review."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ── Browse / List ────────────────────────────────────────────────────

    async def browse(
        self,
        category: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> MarketplaceItemListResponse:
        """Browse published marketplace items, optionally filtered by category."""
        query = select(MarketplaceItem).where(
            MarketplaceItem.status == "published",
            MarketplaceItem.visibility == "public",
        )
        count_query = select(func.count()).select_from(MarketplaceItem).where(
            MarketplaceItem.status == "published",
            MarketplaceItem.visibility == "public",
        )

        if category:
            query = query.where(MarketplaceItem.category == category)
            count_query = count_query.where(MarketplaceItem.category == category)

        # Total count
        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0

        # Fetch page
        query = query.order_by(desc(MarketplaceItem.rating_avg)).offset(offset).limit(limit)
        result = await self.db.execute(query)
        rows = result.scalars().all()

        return MarketplaceItemListResponse(
            items=[MarketplaceItemResponse.model_validate(r) for r in rows],
            total=total,
            limit=limit,
            offset=offset,
        )

    # ── Search ───────────────────────────────────────────────────────────

    async def search(
        self,
        request: MarketplaceSearchRequest,
    ) -> MarketplaceItemListResponse:
        """Full-text search across published marketplace items."""
        query = select(MarketplaceItem).where(
            MarketplaceItem.status == "published",
            MarketplaceItem.visibility == "public",
        )
        count_query = select(func.count()).select_from(MarketplaceItem).where(
            MarketplaceItem.status == "published",
            MarketplaceItem.visibility == "public",
        )

        # Free-text filter on name and description
        if request.query:
            text_filter = or_(
                MarketplaceItem.name.ilike(f"%{request.query}%"),
                MarketplaceItem.description.ilike(f"%{request.query}%"),
                MarketplaceItem.short_description.ilike(f"%{request.query}%"),
            )
            query = query.where(text_filter)
            count_query = count_query.where(text_filter)

        # Category filter
        if request.category:
            query = query.where(MarketplaceItem.category == request.category)
            count_query = count_query.where(MarketplaceItem.category == request.category)

        # Total count
        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0

        # Sorting
        sort_column = getattr(MarketplaceItem, request.sort_by, MarketplaceItem.rating_avg)
        order_fn = desc if request.sort_order == "desc" else asc
        query = query.order_by(order_fn(sort_column))

        # Pagination
        query = query.offset(request.offset).limit(request.limit)
        result = await self.db.execute(query)
        rows = result.scalars().all()

        return MarketplaceItemListResponse(
            items=[MarketplaceItemResponse.model_validate(r) for r in rows],
            total=total,
            limit=request.limit,
            offset=request.offset,
        )

    # ── Get Item ─────────────────────────────────────────────────────────

    async def get_item(self, item_id: str) -> Optional[MarketplaceItemResponse]:
        """Retrieve a single marketplace item by ID."""
        result = await self.db.execute(
            select(MarketplaceItem).where(MarketplaceItem.id == item_id)
        )
        item = result.scalar_one_or_none()
        if item is None:
            return None
        return MarketplaceItemResponse.model_validate(item)

    # ── Install ──────────────────────────────────────────────────────────

    async def install(
        self,
        item_id: str,
        user_id: str,
        request: MarketplaceInstallRequest,
    ) -> MarketplaceInstallResponse:
        """Install a marketplace item. Community limits to 10 active installs per user."""
        # Check item exists and is published
        item_result = await self.db.execute(
            select(MarketplaceItem).where(
                MarketplaceItem.id == item_id,
                MarketplaceItem.status == "published",
            )
        )
        item = item_result.scalar_one_or_none()
        if item is None:
            raise ValueError("Marketplace item not found or not published")

        # Check for existing active installation by this user for this item
        existing_result = await self.db.execute(
            select(MarketplaceInstallation).where(
                MarketplaceInstallation.item_id == item_id,
                MarketplaceInstallation.user_id == user_id,
                MarketplaceInstallation.status == "installed",
            )
        )
        if existing_result.scalar_one_or_none() is not None:
            raise ValueError("Item is already installed")

        # Community install limit
        await self._check_install_limit(user_id)

        installation = MarketplaceInstallation(
            id=str(uuid.uuid4()),
            item_id=item_id,
            user_id=user_id,
            organization_id=request.organization_id,
            version=item.version,
            status="installed",
            installed_at=datetime.utcnow(),
        )
        self.db.add(installation)

        # Increment install count
        item.install_count = (item.install_count or 0) + 1
        await self.db.commit()
        await self.db.refresh(installation)

        return MarketplaceInstallResponse.model_validate(installation)

    async def _check_install_limit(self, user_id: str) -> None:
        """Enforce community per-user installation limit."""
        count_result = await self.db.execute(
            select(func.count()).select_from(MarketplaceInstallation).where(
                MarketplaceInstallation.user_id == user_id,
                MarketplaceInstallation.status == "installed",
            )
        )
        active_count = count_result.scalar() or 0
        if active_count >= COMMUNITY_INSTALL_LIMIT:
            raise ValueError(
                f"Community edition limits installations to {COMMUNITY_INSTALL_LIMIT}. "
                "Upgrade to Business for unlimited installations."
            )

    # ── Uninstall ────────────────────────────────────────────────────────

    async def uninstall(
        self,
        installation_id: str,
        user_id: str,
    ) -> MarketplaceUninstallResponse:
        """Uninstall (soft-delete) an installation."""
        result = await self.db.execute(
            select(MarketplaceInstallation).where(
                MarketplaceInstallation.id == installation_id,
                MarketplaceInstallation.user_id == user_id,
                MarketplaceInstallation.status == "installed",
            )
        )
        installation = result.scalar_one_or_none()
        if installation is None:
            raise ValueError("Installation not found or already uninstalled")

        installation.status = "uninstalled"
        installation.uninstalled_at = datetime.utcnow()

        # Decrement install count on the item
        item_result = await self.db.execute(
            select(MarketplaceItem).where(MarketplaceItem.id == installation.item_id)
        )
        item = item_result.scalar_one_or_none()
        if item and item.install_count > 0:
            item.install_count -= 1

        await self.db.commit()

        return MarketplaceUninstallResponse(
            id=installation.id,
            item_id=installation.item_id,
            status="uninstalled",
            uninstalled_at=installation.uninstalled_at,
        )

    # ── My Installations ─────────────────────────────────────────────────

    async def list_installations(
        self,
        user_id: str,
        status_filter: Optional[str] = "installed",
    ) -> MarketplaceInstallationListResponse:
        """List a user's marketplace installations."""
        query = select(MarketplaceInstallation).where(
            MarketplaceInstallation.user_id == user_id,
        )
        count_query = select(func.count()).select_from(MarketplaceInstallation).where(
            MarketplaceInstallation.user_id == user_id,
        )

        if status_filter:
            query = query.where(MarketplaceInstallation.status == status_filter)
            count_query = count_query.where(MarketplaceInstallation.status == status_filter)

        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0

        result = await self.db.execute(query.order_by(desc(MarketplaceInstallation.installed_at)))
        rows = result.scalars().all()

        return MarketplaceInstallationListResponse(
            installations=[MarketplaceInstallResponse.model_validate(r) for r in rows],
            total=total,
        )

    # ── Reviews ──────────────────────────────────────────────────────────

    async def get_reviews(
        self,
        item_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> MarketplaceReviewListResponse:
        """Get reviews for a marketplace item."""
        count_result = await self.db.execute(
            select(func.count()).select_from(MarketplaceReview).where(
                MarketplaceReview.item_id == item_id,
            )
        )
        total = count_result.scalar() or 0

        result = await self.db.execute(
            select(MarketplaceReview)
            .where(MarketplaceReview.item_id == item_id)
            .order_by(desc(MarketplaceReview.created_at))
            .offset(offset)
            .limit(limit)
        )
        rows = result.scalars().all()

        return MarketplaceReviewListResponse(
            reviews=[MarketplaceReviewResponse.model_validate(r) for r in rows],
            total=total,
        )

    async def create_review(
        self,
        item_id: str,
        user_id: str,
        request: MarketplaceReviewCreate,
    ) -> MarketplaceReviewResponse:
        """Add a review/rating for a marketplace item."""
        # Verify item exists
        item_result = await self.db.execute(
            select(MarketplaceItem).where(MarketplaceItem.id == item_id)
        )
        item = item_result.scalar_one_or_none()
        if item is None:
            raise ValueError("Marketplace item not found")

        # Check if user already reviewed this item
        existing = await self.db.execute(
            select(MarketplaceReview).where(
                MarketplaceReview.item_id == item_id,
                MarketplaceReview.user_id == user_id,
            )
        )
        if existing.scalar_one_or_none() is not None:
            raise ValueError("You have already reviewed this item")

        review = MarketplaceReview(
            id=str(uuid.uuid4()),
            item_id=item_id,
            user_id=user_id,
            rating=request.rating,
            comment=request.comment,
            created_at=datetime.utcnow(),
        )
        self.db.add(review)

        # Recalculate rating
        item.rating_count = (item.rating_count or 0) + 1
        current_total = (item.rating_avg or 0.0) * ((item.rating_count or 1) - 1)
        item.rating_avg = round((current_total + request.rating) / item.rating_count, 2)

        await self.db.commit()
        await self.db.refresh(review)

        return MarketplaceReviewResponse.model_validate(review)
