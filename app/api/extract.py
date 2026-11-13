import logging

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.runnables import Runnable

from app.config import settings
from app.core.llm import get_chat_model
from app.core.prompts import EXTRACT_PROMPT
from app.schemas.extract import AfterSalesTicket, ExtractRequest

logger = logging.getLogger(__name__)
router = APIRouter()


def get_extractor() -> Runnable:
    # 思考开关走配置:换上游只改 .env,不动代码
    extra_body = (
        {"thinking": {"type": settings.extract_thinking}}
        if settings.extract_thinking
        else None
    )
    model = get_chat_model(extra_body=extra_body)
    return EXTRACT_PROMPT | model.with_structured_output(
        AfterSalesTicket, method="function_calling"
    )


@router.post("/api/extract", response_model=AfterSalesTicket)
async def extract(
    req: ExtractRequest, extractor: Runnable = Depends(get_extractor)
) -> AfterSalesTicket:
    try:
        result = await extractor.ainvoke({"text": req.text})
    except Exception as exc:
        logger.exception("结构化提取失败")
        raise HTTPException(status_code=502, detail="上游模型暂时不可用,请稍后重试") from exc
    if result is None:
        # with_structured_output 在模型未产出工具调用时返回 None,不抛异常
        logger.error("结构化提取返回空结果 text=%.50s", req.text)
        raise HTTPException(status_code=502, detail="上游模型未返回可解析的结构化结果,请稍后重试")
    return result
