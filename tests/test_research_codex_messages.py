"""原生交易员消息须无损传入现有纯推理隔离边界。"""

import json

import pytest

from tradingagents.research.codex import prompt_text


def test_text_is_unchanged_and_messages_preserve_order_roles_and_unicode():
    assert prompt_text("原文") == "原文"
    messages = [
        {"role": "system", "content": "仅使用封存证据"},
        {"role": "user", "content": "研究方案\n不得生成订单"},
    ]
    encoded = prompt_text(messages)
    assert json.loads(encoded.split("\n", 1)[1]) == messages
    assert messages[1]["content"].endswith("不得生成订单")


@pytest.mark.parametrize(
    "value",
    [
        [],
        {},
        None,
        [{"role": "tool", "content": "结果"}],
        [{"role": "user", "content": [{"text": "隐藏内容"}]}],
        [{"role": "assistant", "content": "文字", "tool_calls": []}],
        [{"role": "user", "content": "正文", "secret": "额外字段"}],
    ],
)
def test_nontext_or_tool_messages_fail_before_inference(value):
    with pytest.raises(TypeError):
        prompt_text(value)
