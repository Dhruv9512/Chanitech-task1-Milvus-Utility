from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.permissions import AllowAny
from user.models import User
from .tasks import create_and_build_task
from .serializers import BaseKnowledgeSerializer



# Function to validate user from user_id
def validate_user_id(user_id)->object:
    try:
        user = User.objects.get(id=user_id)
        if not user.is_active or user.is_blocked:
            return None
        return user
    except User.DoesNotExist:
        return None

class CreateKBView(APIView):
    permission_classes = [AllowAny]
    def post(self, request):
        try:
            # Use Serializer to validate the input data and Create BaseKnowledge instance
            serializer = BaseKnowledgeSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            obj= serializer.save()
            validated_data = serializer.validated_data
            validated_data["parent_batch_id"]=obj.batch_id
            validated_data["is_openai"]=request.data.get("is_openai")
            
            # If all validations pass, proceed with the KB creation logic
            task = create_and_build_task.delay(**validated_data)

            return Response({
                "status": "success",
                "message": "Knowledge base creation task has been initiated.",
                "task_id": task.id
            }, status=status.HTTP_202_ACCEPTED)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        

