# ============================================================================
# 🎯 TELEGRAM SUBSCRIPTION SALES SYSTEM - ENTERPRISE EDITION
# ============================================================================
# Production-Ready Complete System
# Author: Telegram Shop Development Team
# Version: 3.0 - Enterprise Edition
# Date: 2026
# ============================================================================
# Features:
# ✅ Telegram Bot with Full Features
# ✅ Admin Web Panel
# ✅ User Management
# ✅ Wallet & Balance System
# ✅ Order Management
# ✅ Referral System
# ✅ Subscription Plans
# ✅ Dynamic Buttons
# ✅ Analytics & Statistics
# ✅ Admin Logs
# ============================================================================

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from pathlib import Path

# ============================================================================
# THIRD-PARTY IMPORTS
# ============================================================================

from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean, Enum as SQLEnum,
    ForeignKey, Text, JSON, select, func, desc, and_, or_
)
from sqlalchemy.ext.asyncio import (
    AsyncSession, create_async_engine, async_sessionmaker
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.pool import NullPool

from pydantic import BaseModel, Field, validator
from passlib.context import CryptContext
from jwt import encode, decode, ExpiredSignatureError, InvalidTokenError

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, 
    InlineKeyboardButton, Message
)
from aiogram.enums import ParseMode

import uvicorn
from dotenv import load_dotenv

# ============================================================================
# LOAD ENVIRONMENT & CONFIGURE LOGGING
# ============================================================================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

class Config:
    """System Configuration"""
    
    # Database Configuration
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "sqlite+aiosqlite:///./telegram_shop.db"
    )
    
    # Telegram Bot Configuration
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    if not BOT_TOKEN:
        logger.error("❌ BOT_TOKEN not found in .env")
        sys.exit(1)
    
    BOT_USERNAME = os.getenv("BOT_USERNAME", "your_bot_username")
    CHANNEL_ID = os.getenv("CHANNEL_ID", "-1001234567890")
    CHANNEL_USERNAME = os.getenv("CHANNEL_USERNAME", "your_channel")
    
    # Web API Configuration
    API_SECRET = os.getenv("API_SECRET", "your-secret-key-change-in-production")
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRATION_HOURS = 24
    API_HOST = os.getenv("API_HOST", "127.0.0.1")
    API_PORT = int(os.getenv("API_PORT", 8000))
    
    # Webhook Configuration
    WEBHOOK_URL = os.getenv("WEBHOOK_URL", "http://localhost:8443")
    WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", 8443))
    WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/webhook")
    WEBHOOK_HOST = os.getenv("WEBHOOK_HOST", "127.0.0.1")
    
    # Admin Configuration
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123456")
    
    # Business Logic Configuration
    REFERRAL_REWARD = float(os.getenv("REFERRAL_REWARD", "50000"))
    MIN_DEPOSIT = float(os.getenv("MIN_DEPOSIT", "10000"))
    MAX_DEPOSIT = float(os.getenv("MAX_DEPOSIT", "10000000"))
    
    # Feature Flags
    ENABLE_REFERRAL = os.getenv("ENABLE_REFERRAL", "true").lower() == "true"
    ENABLE_SUBSCRIPTION = os.getenv("ENABLE_SUBSCRIPTION", "true").lower() == "true"
    ENABLE_WALLET = os.getenv("ENABLE_WALLET", "true").lower() == "true"
    
    # Pagination
    ITEMS_PER_PAGE = int(os.getenv("ITEMS_PER_PAGE", "20"))

# ============================================================================
# ENUMS - Business Logic
# ============================================================================

class RoleEnum(str, Enum):
    """User Roles"""
    ADMIN = "admin"
    USER = "user"
    MODERATOR = "moderator"

class TransactionTypeEnum(str, Enum):
    """Transaction Types"""
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    SUBSCRIPTION_PURCHASE = "subscription_purchase"
    REFERRAL_REWARD = "referral_reward"
    ADMIN_ADJUSTMENT = "admin_adjustment"
    SUBSCRIPTION_RENEWAL = "subscription_renewal"

class OrderStatusEnum(str, Enum):
    """Order Status"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class SubscriptionStatusEnum(str, Enum):
    """Subscription Status"""
    ACTIVE = "active"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"

class PaymentMethodEnum(str, Enum):
    """Payment Methods"""
    BANK_TRANSFER = "bank_transfer"
    CRYPTO = "crypto"
    CREDIT_CARD = "credit_card"
    WALLET = "wallet"

# ============================================================================
# DATABASE MODELS
# ============================================================================

Base = declarative_base()

class User(Base):
    """User Model"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    telegram_id = Column(String(20), unique=True, index=True, nullable=False)
    username = Column(String(255), nullable=True, index=True)
    first_name = Column(String(255), nullable=True)
    last_name = Column(String(255), nullable=True)
    phone_number = Column(String(20), nullable=True)
    email = Column(String(255), nullable=True, unique=True, index=True)
    
    # Wallet & Balance
    wallet_balance = Column(Float, default=0.0)
    reserved_balance = Column(Float, default=0.0)  # For pending orders
    
    # User Status
    is_member = Column(Boolean, default=False, index=True)
    is_active = Column(Boolean, default=True, index=True)
    is_banned = Column(Boolean, default=False, index=True)
    ban_reason = Column(Text, nullable=True)
    
    # Referral System
    referred_by = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    is_referral_rewarded = Column(Boolean, default=False)
    total_referrals = Column(Integer, default=0)
    total_referral_earnings = Column(Float, default=0.0)
    
    # User Role & Permissions
    role = Column(SQLEnum(RoleEnum), default=RoleEnum.USER, index=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow, index=True)
    last_login = Column(DateTime, nullable=True)
    
    # Metadata
    metadata = Column(JSON, default={})
    
    # Relations
    transactions = relationship("Transaction", cascade="all, delete-orphan", backref="user")
    orders = relationship("Order", cascade="all, delete-orphan", backref="user")
    subscriptions = relationship("Subscription", cascade="all, delete-orphan", backref="user")
    referrals = relationship("User", remote_side=[referred_by], backref="referrer")

class Transaction(Base):
    """Transaction Model"""
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    
    # Transaction Details
    amount = Column(Float, nullable=False)
    transaction_type = Column(SQLEnum(TransactionTypeEnum), index=True, nullable=False)
    payment_method = Column(SQLEnum(PaymentMethodEnum), nullable=True)
    
    # Transaction Status
    status = Column(String(50), default="completed", index=True)
    description = Column(String(500), nullable=True)
    reference_id = Column(String(100), nullable=True, unique=True)
    
    # Metadata
    metadata = Column(JSON, default={})
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime, nullable=True)

class Order(Base):
    """Order Model"""
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    
    # Order Details
    order_number = Column(String(50), unique=True, index=True, nullable=False)
    amount = Column(Float, nullable=False)
    payment_method = Column(SQLEnum(PaymentMethodEnum), nullable=False)
    
    # Order Status
    status = Column(SQLEnum(OrderStatusEnum), default=OrderStatusEnum.PENDING, index=True)
    
    # Payment Proof
    receipt_file = Column(String(500), nullable=True)
    receipt_url = Column(String(500), nullable=True)
    
    # Admin Actions
    admin_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    admin_notes = Column(Text, nullable=True)
    rejection_reason = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    confirmed_at = Column(DateTime, nullable=True)
    rejected_at = Column(DateTime, nullable=True)

class Subscription(Base):
    """Subscription Model"""
    __tablename__ = "subscriptions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    
    # Subscription Details
    plan_name = Column(String(255), nullable=False)
    plan_id = Column(String(100), nullable=True)
    duration_days = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    features = Column(JSON, default={})
    
    # Subscription Status
    status = Column(SQLEnum(SubscriptionStatusEnum), default=SubscriptionStatusEnum.ACTIVE, index=True)
    
    # Dates
    started_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    auto_renew = Column(Boolean, default=True)
    last_renewed_at = Column(DateTime, nullable=True)
    
    # Metadata
    metadata = Column(JSON, default={})
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

class DynamicButton(Base):
    """Dynamic Button Model"""
    __tablename__ = "dynamic_buttons"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Button Details
    label = Column(String(255), unique=True, index=True, nullable=False)
    url = Column(String(500), nullable=True)
    callback_data = Column(String(255), nullable=True)
    
    # Hierarchy
    parent_id = Column(Integer, ForeignKey("dynamic_buttons.id"), nullable=True)
    
    # Display
    order = Column(Integer, default=0)
    is_active = Column(Boolean, default=True, index=True)
    icon = Column(String(50), nullable=True)
    
    # Metadata
    metadata = Column(JSON, default={})
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relations
    children = relationship("DynamicButton", remote_side=[parent_id])

class SystemSetting(Base):
    """System Settings Model"""
    __tablename__ = "system_settings"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Setting Details
    key = Column(String(255), unique=True, index=True, nullable=False)
    value = Column(Text, nullable=False)
    data_type = Column(String(50), default="string")  # string, int, float, json, bool
    
    # Metadata
    description = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class AdminLog(Base):
    """Admin Activity Log Model"""
    __tablename__ = "admin_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Admin Info
    admin_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    admin_username = Column(String(255), nullable=True)
    
    # Action Details
    action = Column(String(255), nullable=False)
    target_type = Column(String(50), nullable=True)  # user, order, subscription, etc
    target_id = Column(Integer, nullable=True)
    
    # Changes
    old_value = Column(JSON, nullable=True)
    new_value = Column(JSON, nullable=True)
    details = Column(Text, nullable=True)
    
    # Metadata
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

class UserNotification(Base):
    """User Notifications Model"""
    __tablename__ = "user_notifications"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    
    # Notification Details
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    notification_type = Column(String(50), nullable=False)  # info, warning, error, success
    
    # Status
    is_read = Column(Boolean, default=False, index=True)
    read_at = Column(DateTime, nullable=True)
    
    # Metadata
    data = Column(JSON, default={})
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

# ============================================================================
# DATABASE INITIALIZATION & SESSION
# ============================================================================

engine = None
async_session_maker = None

async def init_database():
    """Initialize database and create tables"""
    global engine, async_session_maker
    
    try:
        engine = create_async_engine(
            Config.DATABASE_URL,
            echo=False,
            poolclass=NullPool,
            connect_args={"timeout": 30}
        )
        
        async_session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False
        )
        
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("✅ Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False

async def get_session() -> AsyncSession:
    """Get database session"""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()

# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class LoginRequest(BaseModel):
    """Login request schema"""
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)

class WalletTransactionRequest(BaseModel):
    """Wallet transaction request"""
    user_id: int
    amount: float = Field(..., gt=0)
    transaction_type: TransactionTypeEnum
    payment_method: Optional[PaymentMethodEnum] = None
    description: Optional[str] = None
    reference_id: Optional[str] = None

class OrderApprovalRequest(BaseModel):
    """Order approval request"""
    order_id: int
    status: str
    notes: Optional[str] = None

class ButtonCreateRequest(BaseModel):
    """Create button request"""
    label: str = Field(..., min_length=1)
    url: Optional[str] = None
    callback_data: Optional[str] = None
    parent_id: Optional[int] = None
    order: int = 0

class SettingUpdateRequest(BaseModel):
    """Update setting request"""
    key: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)

class SubscriptionPlanRequest(BaseModel):
    """Create subscription plan"""
    plan_name: str
    duration_days: int = Field(..., gt=0)
    price: float = Field(..., gt=0)
    features: Optional[Dict[str, Any]] = None

# ============================================================================
# SECURITY & AUTHENTICATION
# ============================================================================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """Hash password"""
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    """Verify password"""
    try:
        return pwd_context.verify(plain, hashed)
    except:
        return False

def create_jwt_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=Config.JWT_EXPIRATION_HOURS)
    
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    
    encoded_jwt = encode(
        to_encode,
        Config.API_SECRET,
        algorithm=Config.JWT_ALGORITHM
    )
    return encoded_jwt

def decode_jwt_token(token: str) -> Optional[dict]:
    """Decode JWT token"""
    try:
        payload = decode(
            token,
            Config.API_SECRET,
            algorithms=[Config.JWT_ALGORITHM]
        )
        return payload
    except ExpiredSignatureError:
        logger.warning("Token expired")
        return None
    except InvalidTokenError:
        logger.warning("Invalid token")
        return None
    except Exception as e:
        logger.error(f"Token decode error: {e}")
        return None

async def get_current_admin(
    session: AsyncSession = Depends(get_session),
    token: Optional[str] = None
) -> User:
    """Get current admin user"""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token required"
        )
    
    payload = decode_jwt_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )
    
    result = await session.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    if user.role not in [RoleEnum.ADMIN, RoleEnum.MODERATOR]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return user

# ============================================================================
# REPOSITORIES - USER
# ============================================================================

class UserRepository:
    """User repository for database operations"""
    
    @staticmethod
    async def get_by_telegram_id(session: AsyncSession, telegram_id: str) -> Optional[User]:
        """Get user by telegram ID"""
        result = await session.execute(
            select(User).where(User.telegram_id == telegram_id)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_by_id(session: AsyncSession, user_id: int) -> Optional[User]:
        """Get user by ID"""
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def create(session: AsyncSession, **kwargs) -> User:
        """Create new user"""
        user = User(**kwargs)
        session.add(user)
        await session.flush()
        return user
    
    @staticmethod
    async def update(session: AsyncSession, user_id: int, **kwargs) -> Optional[User]:
        """Update user"""
        user = await UserRepository.get_by_id(session, user_id)
        if user:
            for key, value in kwargs.items():
                if hasattr(user, key):
                    setattr(user, key, value)
            user.updated_at = datetime.utcnow()
            await session.flush()
        return user
    
    @staticmethod
    async def get_all(
        session: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict] = None
    ) -> List[User]:
        """Get all users with optional filters"""
        query = select(User)
        
        if filters:
            if filters.get("is_active") is not None:
                query = query.where(User.is_active == filters["is_active"])
            if filters.get("role"):
                query = query.where(User.role == filters["role"])
            if filters.get("is_banned") is not None:
                query = query.where(User.is_banned == filters["is_banned"])
        
        query = query.order_by(desc(User.created_at)).offset(skip).limit(limit)
        result = await session.execute(query)
        return result.scalars().all()
    
    @staticmethod
    async def get_referrals_count(session: AsyncSession, user_id: int) -> int:
        """Get referral count"""
        result = await session.execute(
            select(func.count(User.id)).where(User.referred_by == user_id)
        )
        return result.scalar() or 0
    
    @staticmethod
    async def get_total_users(session: AsyncSession) -> int:
        """Get total users count"""
        result = await session.execute(select(func.count(User.id)))
        return result.scalar() or 0

# ============================================================================
# REPOSITORIES - TRANSACTION
# ============================================================================

class TransactionRepository:
    """Transaction repository"""
    
    @staticmethod
    async def create(
        session: AsyncSession,
        user_id: int,
        amount: float,
        transaction_type: TransactionTypeEnum,
        payment_method: Optional[PaymentMethodEnum] = None,
        description: Optional[str] = None,
        reference_id: Optional[str] = None
    ) -> Transaction:
        """Create transaction"""
        transaction = Transaction(
            user_id=user_id,
            amount=amount,
            transaction_type=transaction_type,
            payment_method=payment_method,
            description=description,
            reference_id=reference_id
        )
        session.add(transaction)
        await session.flush()
        return transaction
    
    @staticmethod
    async def get_by_user(
        session: AsyncSession,
        user_id: int,
        skip: int = 0,
        limit: int = 50
    ) -> List[Transaction]:
        """Get user transactions"""
        result = await session.execute(
            select(Transaction)
            .where(Transaction.user_id == user_id)
            .order_by(desc(Transaction.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    @staticmethod
    async def get_summary(session: AsyncSession, user_id: int) -> Dict[str, float]:
        """Get transaction summary"""
        result = await session.execute(
            select(Transaction.transaction_type, func.sum(Transaction.amount))
            .where(Transaction.user_id == user_id)
            .group_by(Transaction.transaction_type)
        )
        
        summary = {}
        for tx_type, total in result.all():
            summary[tx_type.value] = total or 0.0
        return summary

# ============================================================================
# REPOSITORIES - ORDER
# ============================================================================

class OrderRepository:
    """Order repository"""
    
    @staticmethod
    async def create(
        session: AsyncSession,
        user_id: int,
        amount: float,
        payment_method: PaymentMethodEnum
    ) -> Order:
        """Create order"""
        order_number = f"ORD-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{user_id}"
        
        order = Order(
            user_id=user_id,
            order_number=order_number,
            amount=amount,
            payment_method=payment_method
        )
        session.add(order)
        await session.flush()
        return order
    
    @staticmethod
    async def get_by_id(session: AsyncSession, order_id: int) -> Optional[Order]:
        """Get order by ID"""
        result = await session.execute(
            select(Order).where(Order.id == order_id)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def update_status(
        session: AsyncSession,
        order_id: int,
        status: str,
        notes: Optional[str] = None
    ) -> Optional[Order]:
        """Update order status"""
        order = await OrderRepository.get_by_id(session, order_id)
        if order:
            order.status = status
            if notes:
                if status == OrderStatusEnum.REJECTED:
                    order.rejection_reason = notes
                else:
                    order.admin_notes = notes
            
            if status == OrderStatusEnum.CONFIRMED:
                order.confirmed_at = datetime.utcnow()
            elif status == OrderStatusEnum.REJECTED:
                order.rejected_at = datetime.utcnow()
            
            order.updated_at = datetime.utcnow()
            await session.flush()
        return order
    
    @staticmethod
    async def get_pending(session: AsyncSession, limit: int = 100) -> List[Order]:
        """Get pending orders"""
        result = await session.execute(
            select(Order)
            .where(Order.status == OrderStatusEnum.PENDING)
            .order_by(Order.created_at)
            .limit(limit)
        )
        return result.scalars().all()
    
    @staticmethod
    async def get_all(
        session: AsyncSession,
        skip: int = 0,
        limit: int = 100
    ) -> List[Order]:
        """Get all orders"""
        result = await session.execute(
            select(Order)
            .order_by(desc(Order.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()

# ============================================================================
# REPOSITORIES - SUBSCRIPTION
# ============================================================================

class SubscriptionRepository:
    """Subscription repository"""
    
    @staticmethod
    async def create(
        session: AsyncSession,
        user_id: int,
        plan_name: str,
        duration_days: int,
        price: float,
        features: Optional[Dict] = None
    ) -> Subscription:
        """Create subscription"""
        expires_at = datetime.utcnow() + timedelta(days=duration_days)
        
        subscription = Subscription(
            user_id=user_id,
            plan_name=plan_name,
            duration_days=duration_days,
            price=price,
            expires_at=expires_at,
            features=features or {}
        )
        session.add(subscription)
        await session.flush()
        return subscription
    
    @staticmethod
    async def get_active_for_user(session: AsyncSession, user_id: int) -> List[Subscription]:
        """Get active subscriptions"""
        result = await session.execute(
            select(Subscription).where(
                and_(
                    Subscription.user_id == user_id,
                    Subscription.status == SubscriptionStatusEnum.ACTIVE,
                    Subscription.expires_at > datetime.utcnow()
                )
            )
        )
        return result.scalars().all()

# ============================================================================
# REPOSITORIES - BUTTON
# ============================================================================

class DynamicButtonRepository:
    """Dynamic button repository"""
    
    @staticmethod
    async def get_all_active(session: AsyncSession) -> List[DynamicButton]:
        """Get all active buttons"""
        result = await session.execute(
            select(DynamicButton)
            .where(DynamicButton.is_active == True)
            .where(DynamicButton.parent_id == None)
            .order_by(DynamicButton.order)
        )
        return result.scalars().all()
    
    @staticmethod
    async def create(
        session: AsyncSession,
        label: str,
        url: Optional[str] = None,
        callback_data: Optional[str] = None,
        parent_id: Optional[int] = None,
        order: int = 0
    ) -> DynamicButton:
        """Create button"""
        button = DynamicButton(
            label=label,
            url=url,
            callback_data=callback_data,
            parent_id=parent_id,
            order=order
        )
        session.add(button)
        await session.flush()
        return button

# ============================================================================
# REPOSITORIES - SETTINGS
# ============================================================================

class SettingRepository:
    """Settings repository"""
    
    @staticmethod
    async def get(session: AsyncSession, key: str) -> Optional[str]:
        """Get setting value"""
        result = await session.execute(
            select(SystemSetting).where(SystemSetting.key == key)
        )
        setting = result.scalar_one_or_none()
        return setting.value if setting else None
    
    @staticmethod
    async def set(session: AsyncSession, key: str, value: str) -> SystemSetting:
        """Set setting value"""
        result = await session.execute(
            select(SystemSetting).where(SystemSetting.key == key)
        )
        setting = result.scalar_one_or_none()
        
        if setting:
            setting.value = value
            setting.updated_at = datetime.utcnow()
        else:
            setting = SystemSetting(key=key, value=value)
            session.add(setting)
        
        await session.flush()
        return setting
    
    @staticmethod
    async def get_all(session: AsyncSession) -> Dict[str, str]:
        """Get all settings"""
        result = await session.execute(select(SystemSetting))
        settings = result.scalars().all()
        return {s.key: s.value for s in settings}

# ============================================================================
# SERVICES - USER SERVICE
# ============================================================================

class UserService:
    """User service"""
    
    @staticmethod
    async def get_or_create_user(
        session: AsyncSession,
        telegram_id: str,
        username: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None
    ) -> User:
        """Get or create user"""
        user = await UserRepository.get_by_telegram_id(session, telegram_id)
        
        if not user:
            user = await UserRepository.create(
                session,
                telegram_id=telegram_id,
                username=username,
                first_name=first_name,
                last_name=last_name
            )
            logger.info(f"✅ New user created: {telegram_id}")
        
        user.last_activity = datetime.utcnow()
        await session.flush()
        return user
    
    @staticmethod
    async def handle_referral(
        session: AsyncSession,
        user_id: int,
        referrer_id: int
    ) -> bool:
        """Handle referral bonus"""
        if not Config.ENABLE_REFERRAL:
            return False
        
        user = await UserRepository.get_by_id(session, user_id)
        
        if not user or user.referred_by or user_id == referrer_id:
            return False
        
        user.referred_by = referrer_id
        await session.flush()
        
        referral_reward = float(
            await SettingRepository.get(session, "referral_reward") or Config.REFERRAL_REWARD
        )
        
        await TransactionRepository.create(
            session,
            referrer_id,
            referral_reward,
            TransactionTypeEnum.REFERRAL_REWARD,
            description=f"Referral reward from user {user_id}"
        )
        
        referrer = await UserRepository.get_by_id(session, referrer_id)
        if referrer:
            referrer.wallet_balance += referral_reward
            referrer.total_referral_earnings += referral_reward
            referrer.total_referrals += 1
        
        logger.info(f"✅ Referral processed: {referrer_id} <- {user_id}")
        return True

# ============================================================================
# SERVICES - WALLET SERVICE
# ============================================================================

class WalletService:
    """Wallet service"""
    
    @staticmethod
    async def add_balance(
        session: AsyncSession,
        user_id: int,
        amount: float,
        transaction_type: TransactionTypeEnum,
        payment_method: Optional[PaymentMethodEnum] = None,
        description: Optional[str] = None
    ) -> bool:
        """Add balance to wallet"""
        if not Config.ENABLE_WALLET:
            return False
        
        if amount <= 0:
            return False
        
        user = await UserRepository.get_by_id(session, user_id)
        if not user:
            return False
        
        user.wallet_balance += amount
        user.updated_at = datetime.utcnow()
        
        await TransactionRepository.create(
            session,
            user_id,
            amount,
            transaction_type,
            payment_method=payment_method,
            description=description
        )
        
        await session.flush()
        logger.info(f"✅ Balance added: {user_id} +{amount}")
        return True
    
    @staticmethod
    async def deduct_balance(
        session: AsyncSession,
        user_id: int,
        amount: float,
        transaction_type: TransactionTypeEnum,
        payment_method: Optional[PaymentMethodEnum] = None,
        description: Optional[str] = None
    ) -> bool:
        """Deduct balance from wallet"""
        if not Config.ENABLE_WALLET:
            return False
        
        if amount <= 0:
            return False
        
        user = await UserRepository.get_by_id(session, user_id)
        if not user or user.wallet_balance < amount:
            return False
        
        user.wallet_balance -= amount
        user.updated_at = datetime.utcnow()
        
        await TransactionRepository.create(
            session,
            user_id,
            amount,
            transaction_type,
            payment_method=payment_method,
            description=description
        )
        
        await session.flush()
        logger.info(f"✅ Balance deducted: {user_id} -{amount}")
        return True
    
    @staticmethod
    async def get_summary(session: AsyncSession, user_id: int) -> Dict[str, Any]:
        """Get wallet summary"""
        user = await UserRepository.get_by_id(session, user_id)
        if not user:
            return {}
        
        transactions = await TransactionRepository.get_by_user(session, user_id)
        summary = await TransactionRepository.get_summary(session, user_id)
        
        return {
            "balance": user.wallet_balance,
            "reserved": user.reserved_balance,
            "available": user.wallet_balance - user.reserved_balance,
            "transactions_count": len(transactions),
            "summary": summary
        }

# ============================================================================
# SERVICES - ORDER SERVICE
# ============================================================================

class OrderService:
    """Order service"""
    
    @staticmethod
    async def create_order(
        session: AsyncSession,
        user_id: int,
        amount: float,
        payment_method: PaymentMethodEnum = PaymentMethodEnum.BANK_TRANSFER
    ) -> Order:
        """Create order"""
        if amount < Config.MIN_DEPOSIT or amount > Config.MAX_DEPOSIT:
            raise ValueError(f"Amount must be between {Config.MIN_DEPOSIT} and {Config.MAX_DEPOSIT}")
        
        order = await OrderRepository.create(
            session,
            user_id,
            amount,
            payment_method
        )
        
        await session.flush()
        logger.info(f"✅ Order created: {order.order_number}")
        return order
    
    @staticmethod
    async def approve_order(
        session: AsyncSession,
        order_id: int,
        admin_id: Optional[int] = None,
        notes: Optional[str] = None
    ) -> bool:
        """Approve order"""
        order = await OrderRepository.get_by_id(session, order_id)
        if not order or order.status != OrderStatusEnum.PENDING:
            return False
        
        success = await WalletService.add_balance(
            session,
            order.user_id,
            order.amount,
            TransactionTypeEnum.DEPOSIT,
            payment_method=order.payment_method,
            description=f"Order {order.order_number} approved"
        )
        
        if success:
            await OrderRepository.update_status(
                session,
                order_id,
                OrderStatusEnum.CONFIRMED,
                notes
            )
            
            order.admin_id = admin_id
            logger.info(f"✅ Order approved: {order.order_number}")
        
        return success
    
    @staticmethod
    async def reject_order(
        session: AsyncSession,
        order_id: int,
        admin_id: Optional[int] = None,
        reason: Optional[str] = None
    ) -> bool:
        """Reject order"""
        order = await OrderRepository.get_by_id(session, order_id)
        if not order:
            return False
        
        await OrderRepository.update_status(
            session,
            order_id,
            OrderStatusEnum.REJECTED,
            reason
        )
        
        order.admin_id = admin_id
        logger.info(f"✅ Order rejected: {order.order_number}")
        return True

# ============================================================================
# BOT - STATE MANAGEMENT
# ============================================================================

class BotStates(StatesGroup):
    """Bot FSM states"""
    waiting_for_amount = State()
    waiting_for_receipt = State()
    waiting_for_confirmation = State()

# ============================================================================
# BOT - HANDLERS
# ============================================================================

class BotHandlers:
    """Telegram bot handlers"""
    
    def __init__(self, session_maker):
        self.session_maker = session_maker
    
    async def start_command(self, message: Message, state: FSMContext):
        """Handle /start command"""
        try:
            async with self.session_maker() as session:
                user = await UserService.get_or_create_user(
                    session,
                    str(message.from_user.id),
                    message.from_user.username,
                    message.from_user.first_name,
                    message.from_user.last_name
                )
                
                # Handle referral
                args = message.text.split()
                if len(args) > 1:
                    try:
                        referrer_id = int(args[1])
                        if referrer_id != user.id:
                            await UserService.handle_referral(session, user.id, referrer_id)
                    except (ValueError, Exception) as e:
                        logger.error(f"Referral error: {e}")
                
                await session.commit()
                
                welcome_msg = f"""
👋 سلام {message.from_user.first_name}

خوش آمدید به سیستم اشتراک ما!

💳 اینجا می‌تونید:
💰 کیف‌پول خود را مدیریت کنید
👥 با ریفرالها درآمد کنید
📦 اشتراک‌های خود را بخرید
                """
                
                keyboard = await self._get_main_keyboard(session)
                await message.answer(welcome_msg, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
        
        except Exception as e:
            logger.error(f"Start command error: {e}")
            await message.answer("❌ خطایی رخ داد. لطفا بعدا امتحان کنید.")
    
    async def wallet_command(self, message: Message):
        """Handle wallet command"""
        try:
            async with self.session_maker() as session:
                user = await UserRepository.get_by_telegram_id(session, str(message.from_user.id))
                
                if not user:
                    await message.answer("❌ کاربر یافت نشد")
                    return
                
                summary = await WalletService.get_summary(session, user.id)
                
                wallet_text = f"""
💰 **کیف‌پول شما**

موجودی: {summary['balance']:,.0f} تومان
موجودی رزرو شده: {summary['reserved']:,.0f} تومان
موجودی قابل استفاده: {summary['available']:,.0f} تومان

📊 خلاصه تراکنش‌ها:
{self._format_summary(summary['summary'])}
                """
                
                keyboard = ReplyKeyboardMarkup(
                    keyboard=[[KeyboardButton(text="🔙 بازگشت")]],
                    resize_keyboard=True
                )
                
                await message.answer(wallet_text, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
        
        except Exception as e:
            logger.error(f"Wallet command error: {e}")
            await message.answer("❌ خطا در دریافت اطلاعات کیف‌پول")
    
    async def referrals_command(self, message: Message):
        """Handle referrals command"""
        try:
            async with self.session_maker() as session:
                user = await UserRepository.get_by_telegram_id(session, str(message.from_user.id))
                
                if not user:
                    await message.answer("❌ کاربر یافت نشد")
                    return
                
                referral_count = await UserRepository.get_referrals_count(session, user.id)
                referral_reward = await SettingRepository.get(
                    session, "referral_reward"
                ) or Config.REFERRAL_REWARD
                
                referrals_text = f"""
👥 **سیستم ریفرالی**

لینک دعوت شما:
`t.me/{Config.BOT_USERNAME}?start={user.id}`

📈 آمار:
• تعداد ریفرال‌ها: {referral_count} نفر
• پاداش هر ریفرال: {float(referral_reward):,.0f} تومان
• درآمد کل: {referral_count * float(referral_reward):,.0f} تومان
                """
                
                keyboard = ReplyKeyboardMarkup(
                    keyboard=[[KeyboardButton(text="🔙 بازگشت")]],
                    resize_keyboard=True
                )
                
                await message.answer(referrals_text, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
        
        except Exception as e:
            logger.error(f"Referrals command error: {e}")
            await message.answer("❌ خطا در دریافت اطلاعات ریفرالی")
    
    async def deposit_command(self, message: Message, state: FSMContext):
        """Handle deposit command"""
        try:
            deposit_text = f"""
💳 **درخواست واریز**

لطفا مبلغ واریزی را وارد کنید:
(حداقل: {Config.MIN_DEPOSIT:,.0f} تومان)
(حداکثر: {Config.MAX_DEPOSIT:,.0f} تومان)
            """
            
            await message.answer(deposit_text, parse_mode=ParseMode.MARKDOWN)
            await state.set_state(BotStates.waiting_for_amount)
        
        except Exception as e:
            logger.error(f"Deposit command error: {e}")
            await message.answer("❌ خطایی رخ داد")
    
    async def amount_handler(self, message: Message, state: FSMContext):
        """Handle amount input"""
        try:
            amount = float(message.text)
            
            if amount < Config.MIN_DEPOSIT or amount > Config.MAX_DEPOSIT:
                await message.answer(
                    f"❌ مبلغ باید بین {Config.MIN_DEPOSIT:,.0f} و {Config.MAX_DEPOSIT:,.0f} تومان باشد"
                )
                return
            
            await state.update_data(amount=amount)
            
            amount_text = f"""
✅ مبلغ: {amount:,.0f} تومان

لطفا فیش پرداخت را ارسال کنید:
            """
            
            keyboard = ReplyKeyboardMarkup(
                keyboard=[[KeyboardButton(text="❌ لغو")]],
                resize_keyboard=True
            )
            
            await message.answer(amount_text, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
            await state.set_state(BotStates.waiting_for_receipt)
        
        except ValueError:
            await message.answer("❌ لطفا عدد صحیح وارد کنید")
        except Exception as e:
            logger.error(f"Amount handler error: {e}")
            await message.answer("❌ خطایی رخ داد")
    
    async def receipt_handler(self, message: Message, state: FSMContext):
        """Handle receipt photo"""
        try:
            if message.photo:
                data = await state.get_data()
                amount = data.get("amount", 0)
                
                async with self.session_maker() as session:
                    user = await UserRepository.get_by_telegram_id(session, str(message.from_user.id))
                    
                    if user:
                        order = await OrderService.create_order(
                            session,
                            user.id,
                            amount,
                            PaymentMethodEnum.BANK_TRANSFER
                        )
                        
                        # Save receipt file ID
                        if message.photo:
                            order.receipt_file = message.photo[-1].file_id
                        
                        await session.commit()
                        
                        success_text = f"""
✅ **سفارش ثبت شد**

📌 شماره سفارش: {order.order_number}
💰 مبلغ: {amount:,.0f} تومان
⏳ وضعیت: در انتظار تایید ادمین

منتظر تایید ادمین باشید...
                        """
                        
                        await message.answer(success_text, parse_mode=ParseMode.MARKDOWN)
            else:
                await message.answer("❌ لطفا یک عکس ارسال کنید")
            
            await state.clear()
        
        except Exception as e:
            logger.error(f"Receipt handler error: {e}")
            await message.answer("❌ خطایی رخ داد")
            await state.clear()
    
    async def _get_main_keyboard(self, session: AsyncSession) -> ReplyKeyboardMarkup:
        """Get main keyboard"""
        buttons = await DynamicButtonRepository.get_all_active(session)
        
        keyboard_buttons = []
        
        for button in buttons:
            keyboard_buttons.append([KeyboardButton(text=button.label)])
        
        keyboard_buttons.extend([
            [KeyboardButton(text="💰 کیف‌پول"), KeyboardButton(text="👥 ریفرالها")],
            [KeyboardButton(text="💳 واریز")]
        ])
        
        return ReplyKeyboardMarkup(keyboard=keyboard_buttons, resize_keyboard=True)
    
    @staticmethod
    def _format_summary(summary: Dict[str, float]) -> str:
        """Format transaction summary"""
        text = ""
        for key, value in summary.items():
            text += f"• {key}: {value:,.0f} تومان\n"
        return text if text else "• بدون تراکنش"

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Telegram Subscription System",
    version="3.0",
    description="Enterprise-grade subscription management system"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API ENDPOINTS - AUTHENTICATION
# ============================================================================

@app.post("/api/auth/login")
async def login(request: LoginRequest, session: AsyncSession = Depends(get_session)):
    """Admin login endpoint"""
    try:
        if request.username == Config.ADMIN_USERNAME and request.password == Config.ADMIN_PASSWORD:
            token = create_jwt_token({"sub": 1, "role": "admin"})
            logger.info(f"✅ Admin login successful")
            return {
                "access_token": token,
                "token_type": "bearer",
                "expires_in": Config.JWT_EXPIRATION_HOURS * 3600
            }
        
        logger.warning(f"❌ Failed login attempt: {request.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ============================================================================
# API ENDPOINTS - USERS
# ============================================================================

@app.get("/api/users")
async def get_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
    admin: User = Depends(get_current_admin)
):
    """Get all users"""
    try:
        users = await UserRepository.get_all(session, skip, limit)
        return [
            {
                "id": u.id,
                "telegram_id": u.telegram_id,
                "username": u.username,
                "first_name": u.first_name,
                "wallet_balance": u.wallet_balance,
                "is_member": u.is_member,
                "is_active": u.is_active,
                "referrals_count": await UserRepository.get_referrals_count(session, u.id),
                "created_at": u.created_at.isoformat(),
                "last_activity": u.last_activity.isoformat()
            }
            for u in users
        ]
    except Exception as e:
        logger.error(f"Get users error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/users/{user_id}")
async def get_user(
    user_id: int,
    session: AsyncSession = Depends(get_session),
    admin: User = Depends(get_current_admin)
):
    """Get user details"""
    try:
        user = await UserRepository.get_by_id(session, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        referral_count = await UserRepository.get_referrals_count(session, user.id)
        transactions = await TransactionRepository.get_by_user(session, user.id)
        
        return {
            "id": user.id,
            "telegram_id": user.telegram_id,
            "username": user.username,
            "first_name": user.first_name,
            "wallet_balance": user.wallet_balance,
            "reserved_balance": user.reserved_balance,
            "referrals_count": referral_count,
            "total_referral_earnings": user.total_referral_earnings,
            "transactions_count": len(transactions),
            "is_member": user.is_member,
            "is_active": user.is_active,
            "is_banned": user.is_banned,
            "created_at": user.created_at.isoformat(),
            "last_activity": user.last_activity.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ============================================================================
# API ENDPOINTS - WALLET
# ============================================================================

@app.post("/api/wallet/transaction")
async def create_transaction(
    request: WalletTransactionRequest,
    session: AsyncSession = Depends(get_session),
    admin: User = Depends(get_current_admin)
):
    """Create wallet transaction"""
    try:
        if request.transaction_type == TransactionTypeEnum.DEPOSIT:
            success = await WalletService.add_balance(
                session,
                request.user_id,
                request.amount,
                request.transaction_type,
                payment_method=request.payment_method,
                description=request.description
            )
        else:
            success = await WalletService.deduct_balance(
                session,
                request.user_id,
                request.amount,
                request.transaction_type,
                payment_method=request.payment_method,
                description=request.description
            )
        
        if not success:
            raise HTTPException(status_code=400, detail="Operation failed")
        
        await session.commit()
        logger.info(f"✅ Transaction created: {request.user_id}")
        return {"status": "success", "message": "Transaction created successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transaction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/wallet/{user_id}")
async def get_wallet_summary(
    user_id: int,
    session: AsyncSession = Depends(get_session),
    admin: User = Depends(get_current_admin)
):
    """Get wallet summary"""
    try:
        summary = await WalletService.get_summary(session, user_id)
        if not summary:
            raise HTTPException(status_code=404, detail="User not found")
        
        return summary
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Wallet summary error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ============================================================================
# API ENDPOINTS - ORDERS
# ============================================================================

@app.get("/api/orders")
async def get_orders(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
    admin: User = Depends(get_current_admin)
):
    """Get all orders"""
    try:
        orders = await OrderRepository.get_all(session, skip, limit)
        return [
            {
                "id": o.id,
                "order_number": o.order_number,
                "user_id": o.user_id,
                "amount": o.amount,
                "status": o.status.value,
                "payment_method": o.payment_method.value,
                "created_at": o.created_at.isoformat(),
                "confirmed_at": o.confirmed_at.isoformat() if o.confirmed_at else None
            }
            for o in orders
        ]
    except Exception as e:
        logger.error(f"Get orders error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/orders/pending")
async def get_pending_orders(
    session: AsyncSession = Depends(get_session),
    admin: User = Depends(get_current_admin)
):
    """Get pending orders"""
    try:
        orders = await OrderRepository.get_pending(session)
        return [
            {
                "id": o.id,
                "order_number": o.order_number,
                "user_id": o.user_id,
                "amount": o.amount,
                "status": o.status.value,
                "payment_method": o.payment_method.value,
                "created_at": o.created_at.isoformat()
            }
            for o in orders
        ]
    except Exception as e:
        logger.error(f"Get pending orders error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/orders/{order_id}/approve")
async def approve_order(
    order_id: int,
    request: OrderApprovalRequest,
    session: AsyncSession = Depends(get_session),
    admin: User = Depends(get_current_admin)
):
    """Approve order"""
    try:
        if request.status == "confirmed":
            success = await OrderService.approve_order(
                session,
                order_id,
                admin_id=admin.id,
                notes=request.notes
            )
        else:
            success = await OrderService.reject_order(
                session,
                order_id,
                admin_id=admin.id,
                reason=request.notes
            )
        
        if not success:
            raise HTTPException(status_code=400, detail="Operation failed")
        
        await session.commit()
        logger.info(f"✅ Order {order_id} {request.status}")
        return {"status": "success", "message": f"Order {request.status} successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Approve order error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ============================================================================
# API ENDPOINTS - STATISTICS
# ============================================================================

@app.get("/api/stats")
async def get_statistics(
    session: AsyncSession = Depends(get_session),
    admin: User = Depends(get_current_admin)
):
    """Get system statistics"""
    try:
        total_users = await session.execute(select(func.count(User.id)))
        total_balance = await session.execute(select(func.sum(User.wallet_balance)))
        total_orders = await session.execute(select(func.count(Order.id)))
        pending_orders = await session.execute(
            select(func.count(Order.id)).where(Order.status == OrderStatusEnum.PENDING)
        )
        completed_orders = await session.execute(
            select(func.count(Order.id)).where(Order.status == OrderStatusEnum.CONFIRMED)
        )
        total_transactions = await session.execute(
            select(func.sum(Transaction.amount))
        )
        
        return {
            "total_users": total_users.scalar() or 0,
            "total_balance": total_balance.scalar() or 0,
            "total_orders": total_orders.scalar() or 0,
            "pending_orders": pending_orders.scalar() or 0,
            "completed_orders": completed_orders.scalar() or 0,
            "total_transactions": total_transactions.scalar() or 0
        }
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ============================================================================
# WEB PAGES - LOGIN
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def login_page():
    """Login page"""
    return """<!DOCTYPE html>
<html dir="rtl" lang="fa">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>پنل مدیریت</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .login-container {
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            width: 100%;
            max-width: 420px;
        }
        .login-container h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 28px;
        }
        .login-container p {
            text-align: center;
            color: #999;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 600;
            font-size: 14px;
        }
        .form-group input {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        .form-group input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        .btn {
            width: 100%;
            padding: 12px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
        }
        .btn:active {
            transform: translateY(0);
        }
        .error {
            color: #e74c3c;
            text-align: center;
            margin-top: 15px;
            display: none;
            background: #ffe5e5;
            padding: 12px;
            border-radius: 8px;
            font-size: 14px;
        }
        .loading {
            display: none;
            text-align: center;
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <h1>🔐 پنل مدیریت</h1>
        <p>سیستم فروش اشتراک تلگرام</p>
        <form id="loginForm">
            <div class="form-group">
                <label for="username">نام کاربری:</label>
                <input type="text" id="username" name="username" value="admin" required autofocus>
            </div>
            <div class="form-group">
                <label for="password">رمز عبور:</label>
                <input type="password" id="password" name="password" value="admin123456" required>
            </div>
            <button type="submit" class="btn">ورود</button>
            <div class="loading" id="loading">درحال ورود...</div>
            <div class="error" id="error"></div>
        </form>
    </div>
    <script>
        document.getElementById('loginForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            const errorDiv = document.getElementById('error');
            const loadingDiv = document.getElementById('loading');
            const btn = document.querySelector('.btn');
            
            try {
                errorDiv.style.display = 'none';
                loadingDiv.style.display = 'block';
                btn.disabled = true;
                
                const response = await fetch('/api/auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    localStorage.setItem('token', data.access_token);
                    loadingDiv.textContent = 'در حال انتقال...';
                    window.location.href = '/dashboard';
                } else {
                    errorDiv.textContent = '❌ نام کاربری یا رمز عبور نادرست است';
                    errorDiv.style.display = 'block';
                    loadingDiv.style.display = 'none';
                    btn.disabled = false;
                }
            } catch (err) {
                errorDiv.textContent = '❌ خطای اتصال';
                errorDiv.style.display = 'block';
                loadingDiv.style.display = 'none';
                btn.disabled = false;
            }
        });
    </script>
</body>
</html>"""

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Dashboard page"""
    return """<!DOCTYPE html>
<html dir="rtl" lang="fa">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>داشبورد</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        .header h1 {
            font-size: 28px;
            color: #333;
        }
        .logout-btn {
            background: #e74c3c;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
        }
        .logout-btn:hover {
            background: #c0392b;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .stat-card h3 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
        }
        .stat-card .number {
            font-size: 36px;
            font-weight: bold;
            color: #333;
        }
        .content {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        .content h2 {
            font-size: 20px;
            margin-bottom: 20px;
            color: #333;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        table th {
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: right;
            font-weight: 600;
        }
        table td {
            padding: 15px;
            border-bottom: 1px solid #e0e0e0;
            text-align: right;
        }
        table tr:hover {
            background: #f9f9f9;
        }
        .action-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            margin-left: 5px;
        }
        .action-btn.reject {
            background: #e74c3c;
        }
        .action-btn:hover {
            opacity: 0.8;
        }
        .status-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }
        .status-pending {
            background: #fff3cd;
            color: #856404;
        }
        .status-confirmed {
            background: #d4edda;
            color: #155724;
        }
        .status-rejected {
            background: #f8d7da;
            color: #721c24;
        }
        .message {
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 15px;
            display: none;
        }
        .message.error {
            background: #ffe5e5;
            color: #e74c3c;
            display: block;
        }
        .message.success {
            background: #e5ffe5;
            color: #27ae60;
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 داشبورد</h1>
            <button class="logout-btn" onclick="logout()">خروج</button>
        </div>
        
        <div class="stats-grid" id="statsGrid"></div>
        
        <div class="content">
            <div id="messageContainer"></div>
            <h2>📦 سفارش‌های در انتظار</h2>
            <table id="ordersTable">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>شماره سفارش</th>
                        <th>کاربر</th>
                        <th>مبلغ</th>
                        <th>وضعیت</th>
                        <th>تاریخ</th>
                        <th>عملیات</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>
    </div>
    
    <script>
        const API_URL = '/api';
        let token = localStorage.getItem('token');
        
        if (!token) {
            window.location.href = '/';
        }
        
        function logout() {
            localStorage.removeItem('token');
            window.location.href = '/';
        }
        
        async function apiCall(endpoint, options = {}) {
            const response = await fetch(API_URL + endpoint, {
                ...options,
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + token,
                    ...options.headers
                }
            });
            return response;
        }
        
        async function loadStats() {
            try {
                const response = await apiCall('/stats');
                const data = await response.json();
                
                const html = `
                    <div class="stat-card">
                        <h3>کل کاربران</h3>
                        <div class="number">${data.total_users}</div>
                    </div>
                    <div class="stat-card">
                        <h3>کل موجودی</h3>
                        <div class="number">${Number(data.total_balance).toLocaleString('fa-IR')}</div>
                    </div>
                    <div class="stat-card">
                        <h3>کل سفارش‌ها</h3>
                        <div class="number">${data.total_orders}</div>
                    </div>
                    <div class="stat-card">
                        <h3>سفارش‌های تایید شده</h3>
                        <div class="number">${data.completed_orders}</div>
                    </div>
                    <div class="stat-card">
                        <h3>سفارش‌های در انتظار</h3>
                        <div class="number">${data.pending_orders}</div>
                    </div>
                `;
                document.getElementById('statsGrid').innerHTML = html;
            } catch (err) {
                console.error('خطا در بارگذاری آمار:', err);
            }
        }
        
        async function loadOrders() {
            try {
                const response = await apiCall('/orders/pending');
                const orders = await response.json();
                
                let html = '';
                if (orders.length === 0) {
                    html = '<tr><td colspan="7" style="text-align: center;">سفارشی در انتظار وجود ندارد</td></tr>';
                } else {
                    orders.forEach(order => {
                        html += `
                            <tr>
                                <td>#${order.id}</td>
                                <td>${order.order_number}</td>
                                <td>${order.user_id}</td>
                                <td>${Number(order.amount).toLocaleString('fa-IR')} تومان</td>
                                <td><span class="status-badge status-${order.status}">${order.status}</span></td>
                                <td>${new Date(order.created_at).toLocaleDateString('fa-IR')}</td>
                                <td>
                                    <button class="action-btn" onclick="approve(${order.id})">✓ تایید</button>
                                    <button class="action-btn reject" onclick="reject(${order.id})">✗ رد</button>
                                </td>
                            </tr>
                        `;
                    });
                }
                
                document.querySelector('#ordersTable tbody').innerHTML = html;
            } catch (err) {
                showError('خطا در بارگذاری سفارش‌ها');
            }
        }
        
        async function approve(id) {
            const notes = prompt('توضیحات (اختیاری):');
            const response = await apiCall(`/orders/${id}/approve`, {
                method: 'POST',
                body: JSON.stringify({
                    order_id: id,
                    status: 'confirmed',
                    notes: notes
                })
            });
            
            if (response.ok) {
                showSuccess('سفارش تایید شد');
                loadOrders();
                loadStats();
            } else {
                showError('خطا در تایید سفارش');
            }
        }
        
        async function reject(id) {
            const notes = prompt('دلیل رد:');
            if (!notes) return;
            
            const response = await apiCall(`/orders/${id}/approve`, {
                method: 'POST',
                body: JSON.stringify({
                    order_id: id,
                    status: 'rejected',
                    notes: notes
                })
            });
            
            if (response.ok) {
                showSuccess('سفارش رد شد');
                loadOrders();
                loadStats();
            } else {
                showError('خطا در رد سفارش');
            }
        }
        
        function showError(msg) {
            const container = document.getElementById('messageContainer');
            container.innerHTML = `<div class="message error">${msg}</div>`;
            setTimeout(() => container.innerHTML = '', 5000);
        }
        
        function showSuccess(msg) {
            const container = document.getElementById('messageContainer');
            container.innerHTML = `<div class="message success">${msg}</div>`;
            setTimeout(() => container.innerHTML = '', 5000);
        }
        
        document.addEventListener('DOMContentLoaded', () => {
            loadStats();
            loadOrders();
            setInterval(loadOrders, 5000);
        });
    </script>
</body>
</html>"""

# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup event"""
    try:
        logger.info("🚀 System starting up...")
        
        success = await init_database()
        if not success:
            logger.error("❌ Failed to initialize database")
            sys.exit(1)
        
        async with async_session_maker() as session:
            # Create default settings
            default_settings = {
                "welcome_message": "👋 خوش آمدید به سیستم اشتراک ما!",
                "referral_reward": str(Config.REFERRAL_REWARD),
                "min_deposit": str(Config.MIN_DEPOSIT),
                "max_deposit": str(Config.MAX_DEPOSIT),
                "system_name": "Telegram Subscription System",
                "system_version": "3.0"
            }
            
            for key, value in default_settings.items():
                existing = await SettingRepository.get(session, key)
                if not existing:
                    await SettingRepository.set(session, key, value)
            
            # Create default buttons
            existing_buttons = await DynamicButtonRepository.get_all_active(session)
            if not existing_buttons:
                default_buttons = [
                    ("🛍️ فروشگاه", None, "shop", None, 0),
                    ("📚 آموزش", None, "learn", None, 1),
                    ("📞 پشتیبانی", None, "support", None, 2),
                ]
                for label, url, callback, parent_id, order in default_buttons:
                    await DynamicButtonRepository.create(
                        session, label, url, callback, parent_id, order
                    )
            
            await session.commit()
        
        logger.info("✅ System initialized successfully")
    
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        sys.exit(1)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    from aiohttp import web
    from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "bot":
            # ============================================================================
            # RUN BOT - POLLING MODE
            # ============================================================================
            async def run_bot():
                """Run Telegram bot with polling"""
                try:
                    await init_database()
                    
                    bot = Bot(token=Config.BOT_TOKEN)
                    dp = Dispatcher()
                    handlers = BotHandlers(async_session_maker)
                    
                    # Register message handlers
                    dp.message.register(
                        handlers.start_command,
                        Command("start")
                    )
                    
                    dp.message.register(
                        handlers.wallet_command,
                        lambda msg: msg.text == "💰 کیف‌پول"
                    )
                    
                    dp.message.register(
                        handlers.referrals_command,
                        lambda msg: msg.text == "👥 ریفرالها"
                    )
                    
                    dp.message.register(
                        handlers.deposit_command,
                        lambda msg: msg.text == "💳 واریز"
                    )
                    
                    dp.message.register(
                        handlers.amount_handler,
                        StateFilter(BotStates.waiting_for_amount)
                    )
                    
                    dp.message.register(
                        handlers.receipt_handler,
                        StateFilter(BotStates.waiting_for_receipt)
                    )
                    
                    logger.info("🤖 Telegram bot started (Polling mode)")
                    logger.info(f"👤 Bot username: @{Config.BOT_USERNAME}")
                    
                    await dp.start_polling(
                        bot,
                        allowed_updates=dp.resolve_used_update_types(),
                        skip_updates=False
                    )
                
                except Exception as e:
                    logger.error(f"Bot error: {e}")
                    sys.exit(1)
            
            asyncio.run(run_bot())
        
        elif command == "web":
            # ============================================================================
            # RUN WEB PANEL
            # ============================================================================
            logger.info(f"🌐 Web panel started on http://{Config.API_HOST}:{Config.API_PORT}")
            logger.info("📱 Open http://localhost:8000 in your browser")
            
            uvicorn.run(
                "telegram_shop_system:app",
                host=Config.API_HOST,
                port=Config.API_PORT,
                reload=False,
                log_level="info"
            )
        
        elif command == "webhook":
            # ============================================================================
            # RUN BOT - WEBHOOK MODE
            # ============================================================================
            async def run_webhook():
                """Run Telegram bot with webhook"""
                try:
                    await init_database()
                    
                    bot = Bot(token=Config.BOT_TOKEN)
                    dp = Dispatcher()
                    handlers = BotHandlers(async_session_maker)
                    
                    # Register message handlers
                    dp.message.register(
                        handlers.start_command,
                        Command("start")
                    )
                    
                    dp.message.register(
                        handlers.wallet_command,
                        lambda msg: msg.text == "💰 کیف‌پول"
                    )
                    
                    dp.message.register(
                        handlers.referrals_command,
                        lambda msg: msg.text == "👥 ریفرالها"
                    )
                    
                    dp.message.register(
                        handlers.deposit_command,
                        lambda msg: msg.text == "💳 واریز"
                    )
                    
                    dp.message.register(
                        handlers.amount_handler,
                        StateFilter(BotStates.waiting_for_amount)
                    )
                    
                    dp.message.register(
                        handlers.receipt_handler,
                        StateFilter(BotStates.waiting_for_receipt)
                    )
                    
                    # Create webhook app
                    aiohttp_app = web.Application()
                    SimpleRequestHandler(
                        dispatcher=dp,
                        bot=bot
                    ).register(aiohttp_app, path=Config.WEBHOOK_PATH)
                    setup_application(aiohttp_app, dp, bot=bot)
                    
                    # Set webhook URL
                    webhook_url = f"{Config.WEBHOOK_URL}{Config.WEBHOOK_PATH}"
                    logger.info(f"🔗 Setting webhook: {webhook_url}")
                    
                    try:
                        await bot.set_webhook(url=webhook_url)
                        logger.info("✅ Webhook configured successfully")
                    except Exception as e:
                        logger.error(f"Webhook configuration error: {e}")
                    
                    # Run server
                    runner = web.AppRunner(aiohttp_app)
                    await runner.setup()
                    site = web.TCPSite(runner, Config.WEBHOOK_HOST, Config.WEBHOOK_PORT)
                    await site.start()
                    
                    logger.info(f"🤖 Telegram bot started (Webhook mode)")
                    logger.info(f"🌐 Server running on {Config.WEBHOOK_HOST}:{Config.WEBHOOK_PORT}")
                    logger.info(f"📍 Webhook URL: {webhook_url}")
                    
                    try:
                        await asyncio.Event().wait()
                    except KeyboardInterrupt:
                        logger.info("⛔ Bot stopped")
                        await runner.cleanup()
                
                except Exception as e:
                    logger.error(f"Webhook error: {e}")
                    sys.exit(1)
            
            asyncio.run(run_webhook())
        
        else:
            logger.error(f"❌ Unknown command: {command}")
            sys.exit(1)
    
    else:
        # Show help
        print("""
╔═══════════════════════════════════════════════════╗
║  🎯 Telegram Subscription System v3.0            ║
║  Enterprise Edition                              ║
╠═══════════════════════════════════════════════════╣
║  Usage:                                          ║
║                                                  ║
║  1. Bot (Polling):                              ║
║     python telegram_shop_system.py bot          ║
║                                                  ║
║  2. Web Panel:                                  ║
║     python telegram_shop_system.py web          ║
║                                                  ║
║  3. Bot (Webhook):                              ║
║     python telegram_shop_system.py webhook      ║
║                                                  ║
╠═══════════════════════════════════════════════════╣
║  Default Credentials:                            ║
║  Username: admin                                 ║
║  Password: admin123456                           ║
╠═══════════════════════════════════════════════════╣
║  Features:                                       ║
║  ✅ User Management                             ║
║  ✅ Wallet System                               ║
║  ✅ Order Management                            ║
║  ✅ Referral System                             ║
║  ✅ Subscription Plans                          ║
║  ✅ Admin Panel                                 ║
║  ✅ Analytics & Reports                         ║
║  ✅ Payment Processing                          ║
║  ✅ Transaction History                         ║
║  ✅ Dynamic Buttons                             ║
╚═══════════════════════════════════════════════════╝
        """)
        sys.exit(0)
