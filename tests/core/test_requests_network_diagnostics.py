from __future__ import annotations

import datetime
import ssl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
import requests
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from puripuly_heart.core.error_messages import format_error_report_for_log, provider_failure_report


def test_real_requests_certificate_failures_preserve_distinct_codes(tmp_path: Path) -> None:
    now = datetime.datetime.now(datetime.UTC)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "diagnostics-private-ca")])

    def builder(subject: x509.Name, key: rsa.RSAPrivateKey) -> x509.CertificateBuilder:
        return (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=1))
            .not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(
                x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False
            )
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
                critical=False,
            )
        )

    ca = builder(issuer, ca_key).add_extension(
        x509.BasicConstraints(ca=True, path_length=None), critical=True
    ).add_extension(
        x509.KeyUsage(
            digital_signature=True, content_commitment=False, key_encipherment=False,
            data_encipherment=False, key_agreement=False, key_cert_sign=True,
            crl_sign=True, encipher_only=False, decipher_only=False,
        ), critical=True
    ).sign(ca_key, hashes.SHA256())
    server_cert = builder(
        x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "wrong.example")]), server_key
    ).add_extension(
        x509.SubjectAlternativeName([x509.DNSName("wrong.example")]), critical=False
    ).sign(ca_key, hashes.SHA256())
    ca_path = tmp_path / "private-ca.pem"
    cert_path = tmp_path / "server.pem"
    key_path = tmp_path / "server-key.pem"
    ca_path.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    cert_path.write_bytes(server_cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(server_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    persisted = []
    try:
        with requests.Session() as session:
            session.trust_env = False
            for verify in (True, str(ca_path)):
                with pytest.raises(requests.exceptions.SSLError) as caught:
                    session.get(f"https://localhost:{server.server_port}/", verify=verify, timeout=3)
                failure = caught.value
                failure.connection_diagnostics = {
                    "transport": "https", "tls_source": "requests_bundle" if verify is True else "explicit_file",
                    "proxy_source": "direct", "tls_backend": "openssl",
                }
                report = provider_failure_report(failure, provider="dashscope", operation="recognize")
                assert report.diagnostics.category == "network"
                assert report.diagnostics.fields["tls_backend"] == "openssl"
                assert "os_errno" not in report.diagnostics.fields
                persisted.append(format_error_report_for_log(report, sink="persisted_logs"))
        assert "tls_verify_code=20" in persisted[0]
        assert "tls_verify_code=62" in persisted[1]
        log_path = tmp_path / "requests-safe.log"
        log_path.write_text("\n".join(persisted), encoding="utf-8")
        content = log_path.read_text(encoding="utf-8")
        assert "localhost" not in content
        assert "wrong.example" not in content
        assert str(tmp_path) not in content
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
