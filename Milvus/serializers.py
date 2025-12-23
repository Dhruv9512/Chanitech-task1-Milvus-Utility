from rest_framework.serializers import ModelSerializer
from .models import BaseKnowledge
from user.models import User
from rest_framework import serializers

# Create a serializer for the BaseKnowledge model
class BaseKnowledgeSerializer(ModelSerializer):
    user_id = serializers.IntegerField(write_only=True)
    is_openai = serializers.BooleanField(required=True)
    class Meta:
        model = BaseKnowledge
        fields = '__all__'
    
    def validate(self, attrs):
        user_id = attrs.get('user_id')
        if user_id:
            try:
                user = User.objects.get(id=user_id)
                if not user.is_active or user.is_blocked:
                    raise serializers.ValidationError("User is inactive or blocked.")
   
                domain = attrs.get('domain')
                batch_id_list = attrs.get('batch_id_list')
                is_openai = attrs.get('is_openai')

                if domain is None or batch_id_list is None or is_openai is None:
                    raise serializers.ValidationError("Missing required fields.")
                return attrs
            except User.DoesNotExist:
                raise serializers.ValidationError("User does not exist.")
    
        return attrs
            

    def create(self, validated_data):
        # 1. Pop non-model fields
        user_id = validated_data.pop('user_id', None)
        is_openai = validated_data.pop('is_openai', False)
    
        input_batch_id = validated_data.get('parent_batch_id', None)
        obj = None
        if input_batch_id:
            obj = BaseKnowledge.objects.filter(parent_batch_id=input_batch_id).first()
            if obj:
                print(f"Resuming task for existing parent_batch_id: {obj.parent_batch_id}")
                return obj
        if not obj:
            obj = BaseKnowledge.objects.create(**validated_data)

        return obj