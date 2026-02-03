"""Community Marketplace database models.

Stores marketplace items (workflows, templates, adapters, agents),
user reviews, and installation records. Community edition limits
installations to 10 per user.
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Text, JSON, DateTime,
    ForeignKey, Index,
)
from models.base import Base


class MarketplaceItem(Base):
    """A publishable marketplace item (workflow, template, adapter, agent)."""

    __tablename__ = "marketplace_items"

    id = Column(String, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    short_description = Column(String(500), nullable=True)
    author_id = Column(String, ForeignKey("users.id"), nullable=False)
    author_name = Column(String(255), nullable=False)
    category = Column(String(50), nullable=False)  # workflow, template, adapter, agent
    item_type = Column(String(100), nullable=True)  # sub-type within category
    version = Column(String(50), nullable=False, server_default="1.0.0")
    tags = Column(JSON, default=[])
    config_schema = Column(JSON, default={})
    install_count = Column(Integer, nullable=False, server_default="0")
    rating_avg = Column(Float, nullable=True)
    rating_count = Column(Integer, nullable=False, server_default="0")
    status = Column(String(50), nullable=False, server_default="draft")  # draft, published, archived
    visibility = Column(String(50), nullable=False, server_default="public")  # public, private, org
    resource_metadata = Column("resource_metadata", JSON, default={})

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_marketplace_items_category", "category"),
        Index("ix_marketplace_items_status", "status"),
        Index("ix_marketplace_items_visibility", "visibility"),
        Index("ix_marketplace_items_author", "author_id"),
        Index("ix_marketplace_items_rating", "rating_avg"),
        {"extend_existing": True},
    )


class MarketplaceReview(Base):
    """A user review/rating for a marketplace item."""

    __tablename__ = "marketplace_reviews"

    id = Column(String, primary_key=True)
    item_id = Column(String, ForeignKey("marketplace_items.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    rating = Column(Integer, nullable=False)  # 1-5
    comment = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_marketplace_reviews_item", "item_id"),
        Index("ix_marketplace_reviews_user", "user_id"),
        {"extend_existing": True},
    )


class MarketplaceInstallation(Base):
    """Tracks installation of a marketplace item by a user/organization."""

    __tablename__ = "marketplace_installations"

    id = Column(String, primary_key=True)
    item_id = Column(String, ForeignKey("marketplace_items.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    organization_id = Column(String, nullable=True)
    version = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False, server_default="installed")  # installed, uninstalled

    installed_at = Column(DateTime, default=datetime.utcnow)
    uninstalled_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_marketplace_installations_item", "item_id"),
        Index("ix_marketplace_installations_user", "user_id"),
        Index("ix_marketplace_installations_org", "organization_id"),
        Index("ix_marketplace_installations_status", "status"),
        {"extend_existing": True},
    )
