from django.db import models
from django.contrib.auth.models import AbstractUser
import uuid
# Create your models here.
class User(AbstractUser): 

    ROLE_CHOICES = (
    ("user", "User"),
    ("admin", "Admin")
    )  
    email = models.EmailField(max_length=254, unique=True)
    contact_number = models.CharField(max_length=15, unique=False, null=True, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    credit_alloted = models.IntegerField(default=15000)
    rag_credit_alloted = models.IntegerField(default=2000)
    visitor_ids = models.JSONField(default=list, blank=True, null=True)  # List of visitor_ids
    fingerprint_ids = models.JSONField(default=list, blank=True, null=True)  # List of fingerprints
    is_active = models.BooleanField(default=False)
    is_guest = models.BooleanField(default=False)
    is_blocked = models.BooleanField(default=False)
    is_admin = models.BooleanField(default=False)
    user_uuid= models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default="user")
    joined_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)  

    def __str__(self):
        return self.username
 