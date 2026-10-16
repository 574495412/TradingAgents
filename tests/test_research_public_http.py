"""验证公开会话不能继承认证、越过路径白名单或在限速后继续请求。"""

import pytest

from tradingagents.dataflows.research.public_http import PublicHTTP


def test_public_session_has_no_implicit_authentication_or_persistent_private_headers():
    client = PublicHTTP(workers=1)
    session = client.session()
    assert session.trust_env is False and session.auth is None
    assert "Authorization" not in session.headers and "X-MBX-APIKEY" not in session.headers
    assert client.session() is session
    client.close()


def test_private_paths_and_authentication_parameters_are_rejected_before_network():
    client = PublicHTTP(workers=1)
    with pytest.raises(ValueError):
        client.request("order", "PUBLIC", {})
    with pytest.raises(ValueError):
        client.request("index", "PUBLIC", {"pair": "PUBLIC", "signature": "not-sent"})
    client.pacer.stop.set()
    assert client.request("clock", None, {})["status"] == "CANCELLED_BEFORE_REQUEST"
    client.close()
