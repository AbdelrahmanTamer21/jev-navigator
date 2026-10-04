from collections import Counter

import pytest

from jev_navigator.judgments.secrets import (
    MASK,
    SecretMasker,
    SecretScanner,
    mask_by_content,
    mask_request,
    refuse_if_secret,
)


def test_repeated_request_text_is_scanned_once_without_retaining_other_requests(monkeypatch):
    masker = SecretMasker()
    mask = masker.mask
    discover = masker.masked_values
    masked = Counter()
    discovered = Counter()

    def observe_mask(self, text):
        masked[text] += 1
        return mask(text)

    def observe_discovery(self, text):
        discovered[text] += 1
        return discover(text)

    monkeypatch.setattr(SecretMasker, "mask", observe_mask)
    monkeypatch.setattr(SecretMasker, "masked_values", observe_discovery)
    reference = 'send("order-hook-4f7a1c")'
    value = {
        "assignment": 'WEBHOOK_TOKEN = "order-hook-4f7a1c"',
        "rows": [{"code": reference} for _ in range(200)],
    }

    result = mask_by_content(value, masker)

    assert result == {
        "assignment": 'WEBHOOK_TOKEN = "[MASKED]"',
        "rows": [{"code": 'send("[MASKED]")'} for _ in range(200)],
    }
    assert discovered[reference] == 1
    assert masked[reference] == 1
    # Without the assignment this ordinary string is not secret-shaped. The previous request's
    # discovered values must not survive as hidden state in a later masking operation.
    assert mask_by_content({"code": reference}, masker) == {"code": reference}


SLACK_TOKEN = "xoxb" + "-123456789012-abcdefghijklmnop"
GITHUB_TOKEN = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
PRIVATE_KEY = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAx9k2\n-----END RSA PRIVATE KEY-----"

SECRET_VALUES = {
    "private key": (f"const key = `{PRIVATE_KEY}`;", "MIIEowIBAAKCAQEAx9k2"),
    "JWT": (
        'const t = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.'
        'dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U";',
        "dozjgNryP4J3jVmNHl0w5N",
    ),
    "GitHub token": (f'auth: "{GITHUB_TOKEN}"', GITHUB_TOKEN),
    "GitHub fine-grained token": (
        'auth: "github_pat_11ABCDEFG0abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP"',
        "github_pat_11ABCDEFG0",
    ),
    "GitLab token": ('const t = "glpat-abcdefghijklmnopqrst";', "glpat-abcdefghijklmnopqrst"),
    "npm token": (
        'const t = "npm_abcdefghijklmnopqrstuvwxyz0123456789";',
        "npm_abcdefghijklmnopqrstuvwxyz0123456789",
    ),
    "OpenAI key": (
        'const k = "sk-proj-abcdefghijklmnopqrstuvwxyz012345";',
        "sk-proj-abcdefghijklmnopqrstuvwxyz012345",
    ),
    "Slack token": (f'const s = "{SLACK_TOKEN}";', SLACK_TOKEN),
    "AWS access key id": ('const id = "AKIAIOSFODNN7EXAMPLE";', "AKIAIOSFODNN7EXAMPLE"),
    "Google API key": (
        'const g = "AIzaSyA1234567890abcdefghijklmnopqrstuv";',
        "AIzaSyA1234567890abcdefghijklmnopqrstuv",
    ),
    "Bearer header": ('headers = {"Authorization": "Bearer abcdefghijklmnop1234"}', "abcdefghijklmnop1234"),
    "quoted secret in JavaScript": (
        'const options = { secret: "s3cr3t-session-value" };',
        "s3cr3t-session-value",
    ),
    "quoted password in Python": ("password = 'hunter2-hunter2'", "hunter2-hunter2"),
    "short quoted password": ("password = 'hunter2'", "hunter2"),
    "quoted JSON key": ('"token": "abcd1234efgh5678"', "abcd1234efgh5678"),
    "bare YAML value": ("api_key: plain-yaml-secret-value", "plain-yaml-secret-value"),
    "env-file value": ("DATABASE_PASSWORD=Pr0dPassw0rd2024", "Pr0dPassw0rd2024"),
    "literal fallback after a reference": (
        'secret: process.env.SESSION_SECRET || "fallback-secret"',
        "fallback-secret",
    ),
    "template literal": ("token = `template-secret-value`", "template-secret-value"),
    "word argument to a signing call": ('token = sign(payload, "hunter2")', "hunter2"),
    "literal argument to a secret-named call": (
        "secret: getSecret('sk-live-signing-0042')",
        "sk-live-signing-0042",
    ),
    "unterminated quoted value": ('password: "unterminated secret', "unterminated secret"),
    "camelCase secret key": ('const authToken = "hunter2";', "hunter2"),
    "high-entropy value under an ordinary name": (
        'const signingKey = "Zq8vT2mN4xR7pL1wK9sD3fH6";',
        "Zq8vT2mN4xR7pL1wK9sD3fH6",
    ),
}

CODE_REFERENCES = [
    "secret: process.env.BETTER_AUTH_SECRET,",
    "token: recipient.token, });",
    "{ id_token: token, id }",
    "const options = { secret: process.env.SESSION_SECRET };",
    'password = os.environ["DB_PASSWORD"]',
    'api_key = settings.get("api_key")',
    '  "token": request.headers.authorization,',
    "const token = await getAccessToken(user);",
    "if (user.email === 'admin@example.com') return 10.0.0.1;",
    "token: Record<string, string>;",
    'password: "${DB_PASSWORD}"',
    "api_key: ${API_KEY}",
    "--token-size: 12px;",
    "signal('SIGTERM')",
    "Returns a Bearer token for the user.",
    "JEV_STATE_TOKEN_LIMIT = 32_000",
    'refuse_if_secret({"prompt": prompt}, {}, scanner)',
    "never print credentials (environment or `~/.config/jvn/env`).",
    "'@aws-sdk/credential-provider-ini': 3.973.15",
    "secretName: heedvane-observability-runtime",
    'SECRETS_DIRECTORY = "secrets"',
    'credentialsMountPath: "/var/run/secrets/google",',
    'if [[ -z "${SLACK_BOT_TOKEN:-}" ]]; then',
    "_render_reports_block(reports, token_budget=...)",
    'secretAnnotation(kind, "name")',
    'requireSecretEnvironment(config, "RUNNER_AUTH_TOKEN", "engine-secrets", "runner-auth-token")',
    "        fencing_token=self.identity.fencing_token,",
    '        hub_token=token or "",',
    "          csrfToken={csrfToken}",
    'this.name = "RunAttemptExecutionClaimConflictError";',
    "secret = {path for file in cited if (path := repo_relative_path(file, repo)) is not None}",
    "# fixture paths are tagged secret: likely fixture by the scanner",
    "// Deprecated env token: an exact match resolves without a round-trip.",
    'RunsRestToken: { in: "header", name: "x-heedvane-runs-rest-token", type: "apiKey" },',
    "existingSecret: { encryptedSecret: Uint8Array; encryptionKeyVersion: number } | null;",
    "secret: {\n  name: AUTH_SECRET_NAME,\n"
    '  items: [{ key: "proxy.htpasswd", path: "proxy.htpasswd", mode: 0o440 }],\n},',
    "emailAndPassword: {\n  enabled: true,\n  // the bounds are the server's copy\n"
    "  minPasswordLength: PASSWORD_MIN_LENGTH,\n},",
    "volumes:\n  - name: runtime\n    secret:\n      secretName: observability-runtime\n"
    "      items:\n        - key: metrics-token\n          path: metrics_token\n",
    "secrets:\n  READ_TOKEN:\n    description: Read-only token for the exact checkout.\n"
    "    required: false\n",
    'WEBHOOK_SECRET="whsec_$(openssl rand -base64 32)"',
]


@pytest.mark.parametrize("code, value", SECRET_VALUES.values(), ids=SECRET_VALUES.keys())
def test_secret_values_are_masked(code: str, value: str) -> None:
    # Act
    masked = SecretMasker().mask(code)

    # Assert
    assert value not in masked
    assert MASK in masked


@pytest.mark.parametrize("code, value", SECRET_VALUES.values(), ids=SECRET_VALUES.keys())
def test_the_scanner_reports_every_secret_value_the_masker_hides(code: str, value: str) -> None:
    # Act
    findings = SecretScanner().findings(code)

    # Assert
    assert findings
    assert SecretScanner().findings(SecretMasker().mask(code)) == []


@pytest.mark.parametrize("code", CODE_REFERENCES)
def test_code_references_reach_jev_byte_identical(code: str) -> None:
    # Act
    masked = SecretMasker().mask(code)

    # Assert
    assert masked == code
    assert SecretScanner().findings(code) == []


def test_only_the_value_is_masked_when_the_key_has_the_same_text() -> None:
    # Act
    masked = SecretMasker().mask('password: "password1234"\nexport const password = "password"')

    # Assert
    assert masked == 'password: "[MASKED]"\nexport const password = "[MASKED]"'


def test_a_short_value_is_masked_where_it_stands_and_nowhere_else_in_the_request() -> None:
    # Arrange
    state = {
        "assignment": "password = 'test'",
        "path": "src/test/login_test.py",
        "test": "a key that names the short value",
    }

    # Act
    masked, questions, values = mask_request(state, {}, SecretMasker())
    refuse_if_secret(masked, questions, SecretScanner(), values)

    # Assert
    assert masked == {
        "assignment": "password = '[MASKED]'",
        "path": "src/test/login_test.py",
        "test": "a key that names the short value",
    }
    assert values == frozenset()


def test_a_long_bare_value_is_masked_everywhere_in_the_request() -> None:
    # Act
    masked = mask_by_content(
        {"config": "api_key: plain-yaml-secret-value", "log": "sent plain-yaml-secret-value upstream"},
        SecretMasker(),
    )

    # Assert
    assert masked == {"config": "api_key: [MASKED]", "log": "sent [MASKED] upstream"}
