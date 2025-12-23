from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User


@admin.register(User)
class CustomUserAdmin(UserAdmin):

    list_display = (
        'username',
        'email',
        'role',
        'is_staff',
        'is_superuser',
        'is_active',
        'is_blocked',
        'joined_at',
    )

    list_filter = (
        'role',
        'is_staff',
        'is_superuser',
        'is_active',
        'is_blocked',
    )

    search_fields = (
        'username',
        'email',
        'contact_number',
    )

    ordering = ('-joined_at',)

    fieldsets = (
        (None, {'fields': ('username', 'password')}),
        ('Personal Info', {
            'fields': (
                'first_name',
                'last_name',
                'email',
                'contact_number',
                'ip_address',
            )
        }),
        ('Permissions', {
            'fields': (
                'is_active',
                'is_staff',
                'is_superuser',
                'is_guest',
                'is_blocked',
                'groups',
                'user_permissions',
            )
        }),
        ('Credits', {
            'fields': (
                'credit_alloted',
                'rag_credit_alloted',
            )
        }),
        ('Tracking', {
            'fields': (
                'visitor_ids',
                'fingerprint_ids',
            )
        }),
        ('Role & Meta', {
            'fields': (
                'role',
                'joined_at',
                'updated_at',
            )
        }),
    )

    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': (
                'username',
                'email',
                'password1',
                'password2',
                'is_staff',
                'is_superuser',
                'is_active',
                'role',
            ),
        }),
    )

    readonly_fields = (
        'user_uuid',
        'joined_at',
        'updated_at',
    )
