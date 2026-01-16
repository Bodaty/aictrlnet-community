"""Multi-Factor Authentication service."""

import pyotp
import qrcode
import io
import base64
import json
import secrets
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from fastapi import HTTPException

from core.config import settings
from core.cache import get_cache
from models.user import User
from core.enforcement_simple import LicenseEnforcer
from core.security import verify_password


class MFAService:
    """Multi-Factor Authentication service with edition-aware features."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        # Initialize Fernet cipher with the encryption key
        # If the key is not a valid Fernet key, generate one from the provided key
        try:
            self.cipher = Fernet(settings.MFA_ENCRYPTION_KEY.encode())
        except ValueError:
            # Generate a proper Fernet key from the settings key
            import hashlib
            import base64
            # Use SHA256 to get 32 bytes from any input
            key_hash = hashlib.sha256(settings.MFA_ENCRYPTION_KEY.encode()).digest()
            # Encode to base64 for Fernet
            fernet_key = base64.urlsafe_b64encode(key_hash)
            self.cipher = Fernet(fernet_key)
        self.license_enforcer = LicenseEnforcer(db)
        
    async def get_user(self, user_id: str) -> User:
        """Get user by ID."""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(404, "User not found")
        return user
    
    async def _can_use_mfa(self, user: User) -> bool:
        """Check if user's edition supports MFA."""
        # Get tenant info to check edition
        tenant_info = await self.license_enforcer._get_tenant_info(user.tenant_id)
        
        # MFA is available for Community Cloud ($49/month) and above
        # In simple enforcement, we only have basic editions
        if tenant_info["edition"] == "community":
            return False  # Free Community edition doesn't have MFA
        
        return True  # Business and Enterprise editions have MFA
    
    async def _supports_multiple_devices(self, user: User) -> bool:
        """Check if user's edition supports multiple MFA devices."""
        tenant_info = await self.license_enforcer._get_tenant_info(user.tenant_id)
        
        # Multiple devices only for Business and Enterprise
        return tenant_info["edition"] in ["business_starter", "business_growth", "business_scale", "enterprise"]
    
    async def _get_backup_code_count(self, user: User) -> int:
        """Get number of backup codes based on edition."""
        tenant_info = await self.license_enforcer._get_tenant_info(user.tenant_id)
        
        # Community Cloud gets 8, Business/Enterprise get 12
        if tenant_info["edition"] == "community":
            return 8
        return 12
    
    async def init_mfa_enrollment(
        self,
        user_id: str,
        device_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Initialize MFA enrollment for a user."""
        
        # Check edition limits
        user = await self.get_user(user_id)
        if not await self._can_use_mfa(user):
            raise HTTPException(403, "MFA not available in your edition. Upgrade to Community Cloud or higher.")
        
        # Check if already enrolled
        if user.mfa_enabled and not await self._supports_multiple_devices(user):
            raise HTTPException(400, "MFA already enabled. Multiple devices not supported in your edition.")
        
        # Generate new secret
        secret = pyotp.random_base32()
        
        # Get backup code count based on edition
        backup_code_count = await self._get_backup_code_count(user)
        
        # Create provisional enrollment
        enrollment = {
            "secret": secret,
            "qr_code": await self._generate_qr_code(user.email, secret),
            "manual_entry_key": secret,
            "backup_codes": await self._generate_backup_codes(backup_code_count)
        }
        
        # Store temporarily in cache
        cache_key = f"mfa_enrollment:{user_id}"
        cache = await get_cache()
        await cache.set(cache_key, enrollment, expire=600)  # 10 min expiry
        
        return {
            "qr_code": enrollment["qr_code"],
            "manual_entry_key": enrollment["manual_entry_key"],
            "expires_in": 600
        }
    
    async def complete_mfa_enrollment(
        self,
        user_id: str,
        verification_code: str,
        device_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Complete MFA enrollment by verifying TOTP code."""
        
        # Get enrollment data from cache
        cache_key = f"mfa_enrollment:{user_id}"
        cache = await get_cache()
        enrollment = await cache.get(cache_key)
        
        if not enrollment:
            raise HTTPException(400, "Enrollment expired or not found")
        
        # Verify the code
        totp = pyotp.TOTP(enrollment["secret"])
        if not totp.verify(verification_code, valid_window=1):
            raise HTTPException(400, "Invalid verification code")
        
        # Save to database
        user = await self.get_user(user_id)
        
        # Encrypt sensitive data
        encrypted_secret = self._encrypt(enrollment["secret"])
        encrypted_backup_codes = self._encrypt(
            json.dumps(enrollment["backup_codes"])
        )
        
        # Update user
        user.mfa_enabled = True
        user.mfa_secret = encrypted_secret
        user.mfa_backup_codes = encrypted_backup_codes
        user.mfa_enrolled_at = datetime.utcnow()
        
        # For Business/Enterprise, create device entry
        if await self._supports_multiple_devices(user):
            try:
                from aictrlnet_business.models.mfa import MFADevice
                import uuid

                device = MFADevice(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    device_name=device_name or "Default Device",
                    device_type="totp",
                    secret_encrypted=encrypted_secret,
                    is_primary=True
                )
                self.db.add(device)
            except ImportError:
                pass  # Business module not available
        
        # Audit log
        await self._audit_log(user_id, "enabled", success=True)
        
        await self.db.commit()
        await cache.delete(cache_key)
        
        return {
            "backup_codes": enrollment["backup_codes"],
            "message": "MFA successfully enabled"
        }
    
    async def verify_mfa_code(
        self,
        user_id: str,
        code: str,
        allow_backup_code: bool = True
    ) -> Dict[str, Any]:
        """Verify MFA code during login."""
        
        user = await self.get_user(user_id)
        
        if not user.mfa_enabled:
            return {"valid": True, "mfa_required": False}
        
        # Try TOTP first
        if len(code) == 6 and code.isdigit():
            secret = self._decrypt(user.mfa_secret)
            totp = pyotp.TOTP(secret)
            
            if totp.verify(code, valid_window=1):
                user.mfa_last_used_at = datetime.utcnow()
                await self._audit_log(user_id, "verified", success=True)
                await self.db.commit()
                return {"valid": True, "mfa_required": True}
        
        # Try backup code if allowed
        if allow_backup_code and len(code) == 8:
            if user.mfa_backup_codes:
                backup_codes = json.loads(self._decrypt(user.mfa_backup_codes))
                
                if code in backup_codes:
                    # Remove used code
                    backup_codes.remove(code)
                    user.mfa_backup_codes = self._encrypt(json.dumps(backup_codes))
                    
                    await self._audit_log(
                        user_id, 
                        "verified", 
                        success=True,
                        details={"method": "backup_code"}
                    )
                    await self.db.commit()
                    
                    return {
                        "valid": True, 
                        "mfa_required": True,
                        "backup_code_used": True,
                        "remaining_codes": len(backup_codes)
                    }
        
        # Invalid code
        await self._audit_log(user_id, "failed", success=False)
        return {"valid": False, "mfa_required": True}
    
    async def disable_mfa(
        self,
        user_id: str,
        password: Optional[str] = None,
        admin_override: bool = False,
        performed_by: Optional[str] = None
    ) -> Dict[str, str]:
        """Disable MFA for a user."""
        
        user = await self.get_user(user_id)
        
        # Verify password unless admin override
        if not admin_override and password:
            if not verify_password(password, user.hashed_password):
                raise HTTPException(401, "Invalid password")
        
        # Clear MFA fields
        user.mfa_enabled = False
        user.mfa_secret = None
        user.mfa_backup_codes = None
        user.mfa_enrolled_at = None
        
        # Delete devices if Business/Enterprise
        if await self._supports_multiple_devices(user):
            try:
                from aictrlnet_business.models.mfa import MFADevice

                await self.db.execute(
                    delete(MFADevice).where(MFADevice.user_id == user_id)
                )
            except ImportError:
                pass  # Business module not available
        
        await self._audit_log(
            user_id, 
            "disabled", 
            success=True,
            performed_by=performed_by if admin_override else None
        )
        
        await self.db.commit()
        
        return {"message": "MFA successfully disabled"}
    
    async def regenerate_backup_codes(
        self,
        user_id: str,
        password: str
    ) -> Dict[str, Any]:
        """Regenerate backup codes for a user."""
        
        user = await self.get_user(user_id)
        
        if not user.mfa_enabled:
            raise HTTPException(400, "MFA not enabled")
        
        # Verify password
        if not verify_password(password, user.hashed_password):
            raise HTTPException(401, "Invalid password")
        
        # Generate new codes
        backup_code_count = await self._get_backup_code_count(user)
        new_codes = await self._generate_backup_codes(backup_code_count)
        
        # Encrypt and save
        user.mfa_backup_codes = self._encrypt(json.dumps(new_codes))
        
        await self._audit_log(user_id, "backup_codes_regenerated", success=True)
        await self.db.commit()
        
        return {
            "recovery_codes": new_codes,
            "generated_at": datetime.utcnow()
        }
    
    async def get_mfa_status(self, user_id: str) -> Dict[str, Any]:
        """Get MFA status for a user."""
        user = await self.get_user(user_id)
        
        response = {
            "mfa_enabled": user.mfa_enabled,
            "enrolled_at": user.mfa_enrolled_at,
            "last_used_at": user.mfa_last_used_at
        }
        
        # Business/Enterprise features
        if await self._supports_multiple_devices(user):
            try:
                from aictrlnet_business.models.mfa import MFADevice

                result = await self.db.execute(
                    select(MFADevice).where(MFADevice.user_id == user_id)
                )
                devices = result.scalars().all()

                response["devices"] = [
                    {
                        "id": str(device.id),
                        "device_name": device.device_name,
                        "device_type": device.device_type,
                        "is_primary": device.is_primary,
                        "last_used_at": device.last_used_at,
                        "created_at": device.created_at
                    }
                    for device in devices
                ]
            except ImportError:
                pass  # Business module not available
        
        return response
    
    async def _generate_qr_code(self, email: str, secret: str) -> str:
        """Generate QR code for TOTP enrollment."""
        
        provisioning_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=email,
            issuer_name='AICtrlNet'
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    
    async def _generate_backup_codes(self, count: int = 8) -> List[str]:
        """Generate backup recovery codes."""
        
        codes = []
        for _ in range(count):
            # Generate 8-character alphanumeric codes
            code = ''.join(secrets.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') 
                          for _ in range(8))
            codes.append(code)
        
        return codes
    
    def _encrypt(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.cipher.encrypt(data.encode()).decode()
    
    def _decrypt(self, data: str) -> str:
        """Decrypt sensitive data."""
        return self.cipher.decrypt(data.encode()).decode()
    
    async def _audit_log(
        self,
        user_id: str,
        action: str,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
        performed_by: Optional[str] = None
    ) -> None:
        """Log MFA actions for audit trail."""
        
        # For Business/Enterprise editions, create audit log entry
        user = await self.get_user(user_id)
        if await self._supports_multiple_devices(user):
            try:
                from aictrlnet_business.models.mfa import MFAAuditLog
                import uuid

                audit_entry = MFAAuditLog(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    action=action,
                    success=success,
                    performed_by=performed_by,
                    failure_reason=details.get("failure_reason") if details else None
                )
                self.db.add(audit_entry)
                # Don't commit here, let caller handle transaction
            except ImportError:
                pass  # Business module not available