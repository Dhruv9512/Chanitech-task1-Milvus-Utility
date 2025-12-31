from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from Milvus.models import BaseKnowledge
from .KB_Utility import MilvusKnowledgeBaseBuilder
from rest_framework.permissions import AllowAny

# Create a View For Create and Build Knowledge Base
class CreateAndBuildKB(APIView):
    permission_classes=[AllowAny]
    def post(self, request,batch_id):
        query=request.data.get("query")
        base_instance = BaseKnowledge.objects.filter(batch_id=batch_id).first()
        if not base_instance:
            return Response({"message": "Base knowledge not found"}, status=status.HTTP_404_NOT_FOUND)
        
        if base_instance.embeded_model_name == "intfloat/multilingual-e5-large":
            is_openai = True
        else:
            is_openai = False

        milvus_builder = MilvusKnowledgeBaseBuilder(
            domain=base_instance.domain,
            batch_id=str(batch_id),
            is_openai=is_openai,
            query=query
        )
        result = milvus_builder.build()

        return Response({"message": result}, status=status.HTTP_200_OK)
