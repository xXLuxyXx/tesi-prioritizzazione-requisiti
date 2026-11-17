import os
import re
import time
import json
import random
import platform
import warnings
warnings.filterwarnings("ignore")

import psutil
import joblib
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, average_precision_score
)
from sklearn.metrics.pairwise import cosine_similarity
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

# ==============================================================================
# 0. RIPRODUCIBILITÀ E SPECIFICHE HARDWARE (Sez. 4.1.1)
# ==============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("=" * 80)
print("AVVIO PIPELINE SPERIMENTALE E ABLATION STUDY (TESI LUCA ADESSO)")
print("=" * 80)

hw_specs = {
    "os": f"{platform.system()} {platform.release()}",
    "cpu": platform.processor() or platform.machine(),
    "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N.A. (CPU execution)",
    "python_version": platform.python_version(),
    "seed": SEED
}
print("Specifiche Ambiente:", hw_specs)

# ==============================================================================
# 1. PRE-PROCESSING DOMAIN-AGNOSTIC (Sez. 3.1.1)
# ==============================================================================
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

DEONTIC_AND_NEG = {
    "must", "shall", "should", "may", "might", "can", "could", "will", "would",
    "not", "no", "never", "none", "cannot", "without", "before", "after", "within"
}
CUSTOM_STOPWORDS = nlp.Defaults.stop_words - DEONTIC_AND_NEG

ACRONYM_MAP = {
    r"\bauth\b": "authentication",
    r"\bdb\b": "database",
    r"\bperf\b": "performance",
    r"\breq\b": "requirement",
    r"\bconfig\b": "configuration",
    r"\bui\b": "user interface",
    r"\bapi\b": "application programming interface",
    r"\bsec\b": "security"
}

def clean_requirement_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[?[A-Z]{2,6}[-_]\d+\]?:?", " ", text)
    text = re.sub(r"#\d+", " ", text)
    text = re.sub(r"\b\d+(\.\d+)?\s*(ms|sec|seconds|minutes|min|hours|s)\b", " <TIME_CONSTRAINT> ", text, flags=re.I)
    text = re.sub(r"\b\d+(\.\d+)?\s*%", " <PERCENTAGE_THRESHOLD> ", text)
    text = re.sub(r"\b\d+(\.\d+)?\s*(mb|gb|kb|requests/s|req/s|tps)\b", " <CAPACITY_THRESHOLD> ", text, flags=re.I)
    text = text.lower()
    for pat, repl in ACRONYM_MAP.items():
        text = re.sub(pat, repl, text)
    text = re.sub(r"[^a-z0-9_<>\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    doc = nlp(text)
    tokens = []
    for tok in doc:
        if tok.text.startswith("<") and tok.text.endswith(">"):
            tokens.append(tok.text)
        elif tok.text in DEONTIC_AND_NEG:
            tokens.append(tok.text)
        elif tok.text not in CUSTOM_STOPWORDS and len(tok.text) > 1:
            tokens.append(tok.lemma_)
    return " ".join(tokens)

UI_TO_DATASET = {
    "functional": "fr",
    "non-functional": "nfr",
    "non functional": "nfr",
    "quality": "qr"
}

def normalize_cat_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Type", "Change", "Urgency", "Origin"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().replace(UI_TO_DATASET)
    return df

# ==============================================================================
# 2. CARICAMENTO DATASET E PARTIZIONAMENTO STRATIFICATO ANTI-LEAKAGE (Sez. 4.1.2)
# ==============================================================================
df_full = normalize_cat_df(pd.read_csv("dataset/full_requirements.csv"))
df_full["Priority"] = df_full["Priority"].str.strip().str.lower()

def get_dist_stats(df_sub: pd.DataFrame) -> dict:
    vc = df_sub["Priority"].value_counts()
    n = len(df_sub)
    h, m, l = int(vc.get("high", 0)), int(vc.get("medium", 0)), int(vc.get("low", 0))
    ir = round(max(h, m, l) / max(min(h, m, l), 1), 2)
    return {
        "N": n,
        "high": h, "high_pct": round(100 * h / n, 2),
        "medium": m, "medium_pct": round(100 * m / n, 2),
        "low": l, "low_pct": round(100 * l / n, 2),
        "IR": ir
    }

train_base, test_lab = train_test_split(
    df_full, test_size=0.20, stratify=df_full["Priority"], random_state=SEED
)
train_base = train_base.reset_index(drop=True)
test_lab = test_lab.reset_index(drop=True)

# ==============================================================================
# 3. RUNTIME BENCHMARK TEST SET DISGIUNTO (USER STORY AGILE MODERNE)
# ==============================================================================
RUNTIME_STORIES = [
    # --- HIGH PRIORITY (20 istanze critiche moderne - allineate alla logica ROI di PROMISE) ---
    ("nfr", "As a security architect, I want all REST API endpoints to validate signed JWT access tokens via OAuth2 within 15 ms, so that unauthorized consumers are blocked before accessing tenant data.", "addition", "soon", "organization", 2, 2, 3, "high"),
    ("nfr", "As a DevOps engineer, I want automated failover across multi-AZ Kubernetes clusters with RTO under 30 seconds, so that critical payment microservices remain available during node outages.", "addition", "soon", "project vision", 3, 3, 3, "high"),
    ("fr", "As a compliance officer, I want personal identifiable information (PII) to be encrypted at rest using AES-256 and purged upon GDPR erasure requests, so that the platform avoids regulatory penalties.", "addition", "soon", "customer", 2, 2, 2, "high"),
    ("nfr", "As a platform engineer, I want distributed rate limiting and circuit breakers configured on the API Gateway, so that cascading failures and DDoS bursts do not degrade core checkout services.", "addition", "soon", "organization", 2, 2, 2, "high"),
    ("fr", "As an enterprise administrator, I want mandatory SAML 2.0 Single Sign-On and role-based access control (RBAC) enforcement across microservices, so that privilege escalation vulnerabilities are prevented.", "addition", "soon", "customer", 2, 2, 3, "high"),
    ("nfr", "As a database reliability engineer, I want distributed ACID transaction consistency using the Saga pattern with idempotency keys, so that duplicate payment charges never occur during network retries.", "modification", "soon", "developer knowledge", 3, 3, 3, "high"),
    ("nfr", "As a security engineer, I want all container images scanned in the CI/CD pipeline and critical CVE vulnerabilities to block production deployment, so that compromised dependencies never reach runtime.", "addition", "soon", "organization", 3, 3, 4, "high"),
    ("fr", "As a healthcare practitioner, I want immutable audit logs capturing every read and write operation on patient records with cryptographic hashing, so that forensic traceability is guaranteed.", "addition", "soon", "customer", 2, 2, 2, "high"),
    ("nfr", "As a site reliability engineer, I want p99 latency for the real-time fraud scoring microservice to remain below 80 ms under 5000 requests/s, so that checkout throughput is not bottlenecked.", "modification", "soon", "market", 2, 2, 3, "high"),
    ("fr", "As a mobile banking user, I want biometric MFA challenge verification before executing wire transfers exceeding threshold limits, so that account takeover attacks are neutralized.", "addition", "soon", "customer", 3, 3, 2, "high"),
    ("nfr", "As a cloud architect, I want mutual TLS (mTLS) enforced across all service-to-service traffic inside the service mesh, so that lateral movement inside the cluster is prevented.", "addition", "soon", "organization", 2, 2, 2, "high"),
    ("fr", "As a payment gateway integrator, I want HMAC-SHA256 signature verification on all incoming webhook payloads, so that spoofed transaction confirmation events are rejected immediately.", "addition", "soon", "developer knowledge", 3, 3, 4, "high"),
    ("nfr", "As an operations lead, I want automated point-in-time database backups replicated cross-region every 5 minutes with zero data loss (RPO=0), so that catastrophic corruption can be recovered.", "addition", "soon", "project vision", 2, 2, 2, "high"),
    ("fr", "As a security operations analyst, I want automatic session token revocation across all active devices immediately upon password reset or suspicious login detection, so that hijacked sessions terminate.", "modification", "soon", "organization", 3, 3, 4, "high"),
    ("nfr", "As a backend architect, I want strict schema validation and parameterized query enforcement on GraphQL resolvers, so that injection and query-depth DoS attacks are eliminated.", "addition", "soon", "developer knowledge", 3, 3, 2, "high"),
    ("fr", "As a fintech compliance auditor, I want real-time AML transaction screening integrated into the settlement pipeline before funds are released, so that sanctioned entities are blocked.", "addition", "soon", "customer", 3, 3, 3, "high"),
    ("nfr", "As a platform SRE, I want OpenTelemetry distributed tracing and automated PagerDuty alerting on error budget burn rates above 2%, so that production incidents are mitigated before SLA breach.", "addition", "soon", "organization", 3, 3, 4, "high"),
    ("fr", "As an API consumer, I want short-lived refresh token rotation with automatic reuse detection, so that stolen OAuth2 credentials cannot be replayed by attackers.", "modification", "soon", "developer knowledge", 3, 3, 2, "high"),
    ("nfr", "As a systems engineer, I want zero-downtime blue-green database migrations with backward-compatible schemas, so that continuous deployment never interrupts live customer traffic.", "addition", "soon", "project vision", 2, 2, 3, "high"),
    ("fr", "As a tenant administrator, I want strict row-level tenant isolation enforced at the database connection pool layer, so that cross-tenant data leakage is mathematically impossible.", "addition", "soon", "customer", 2, 2, 2, "high"),

    # --- MEDIUM PRIORITY (35 istanze moderne) ---
    ("fr", "As a product manager, I want an interactive analytics dashboard displaying sprint velocity and cycle time metrics, so that team capacity can be forecasted more accurately.", "addition", "soon", "organization", 2, 4, 4, "medium"),
    ("nfr", "As a backend developer, I want Redis distributed caching added for read-heavy catalog endpoints, so that database CPU utilization decreases by 40% during peak hours.", "addition", "soon", "developer knowledge", 2, 3, 6, "medium"),
    ("fr", "As an end user, I want real-time push notifications via WebSockets when my support ticket status changes, so that I do not need to manually refresh the portal.", "addition", "soon", "market", 2, 4, 6, "medium"),
    ("fr", "As a customer support agent, I want full-text search with faceted filtering across historical order records, so that customer inquiries are resolved faster.", "modification", "soon", "customer", 2, 4, 4, "medium"),
    ("nfr", "As a QA automation engineer, I want ephemeral preview environments spun up automatically for every pull request in GitHub Actions, so that integration tests run before merge.", "addition", "soon", "developer knowledge", 2, 4, 6, "medium"),
    ("fr", "As a workspace owner, I want to export monthly billing invoices and usage telemetry in CSV and PDF formats, so that our accounting team can reconcile cloud expenses.", "addition", "soon", "customer", 2, 3, 6, "medium"),
    ("fr", "As a mobile app user, I want offline-first local synchronization for draft notes with conflict resolution upon reconnection, so that I can work during flights.", "addition", "later", "market", 3, 3, 4, "medium"),
    ("nfr", "As a frontend architect, I want code splitting and lazy loading configured for the React micro-frontends, so that initial bundle size stays below 250 KB.", "modification", "soon", "developer knowledge", 2, 3, 6, "medium"),
    ("fr", "As a team lead, I want Slack and Microsoft Teams webhook integrations for deployment status notifications, so that developers stay informed of release pipelines.", "addition", "later", "organization", 1, 3, 5, "medium"),
    ("fr", "As a registered customer, I want to manage multiple shipping addresses and set a default payment card in my profile settings, so that repeat checkouts take fewer steps.", "modification", "soon", "customer", 2, 3, 6, "medium"),
    ("nfr", "As a DevOps specialist, I want structured JSON logging shipped to Elasticsearch with 30-day retention policies, so that microservice debugging across containers is streamlined.", "addition", "soon", "organization", 2, 3, 6, "medium"),
    ("fr", "As a content editor, I want scheduled publishing workflows for marketing landing pages with preview links, so that campaigns launch automatically at midnight.", "addition", "later", "market", 2, 4, 6, "medium"),
    ("fr", "As an API partner, I want sandbox API keys and interactive Swagger/OpenAPI documentation, so that our developers can test webhook integrations safely.", "addition", "soon", "market", 1, 3, 6, "medium"),
    ("nfr", "As a mobile engineer, I want adaptive image compression and CDN edge caching for user-uploaded media, so that feed scrolling remains smooth on 4G networks.", "addition", "later", "developer knowledge", 2, 3, 6, "medium"),
    ("fr", "As a store manager, I want automated low-stock threshold alerts sent via email when inventory drops below reorder levels, so that stockouts are minimized.", "addition", "soon", "customer", 2, 3, 5, "medium"),
    ("fr", "As a data analyst, I want nightly ETL pipelines syncing operational events into the Snowflake data warehouse, so that BI reports reflect previous-day activity.", "addition", "later", "organization", 3, 4, 4, "medium"),
    ("nfr", "As a platform developer, I want feature flag toggles integrated via LaunchDarkly SDK, so that new capabilities can be rolled out gradually to 10% of users.", "addition", "soon", "developer knowledge", 2, 3, 6, "medium"),
    ("fr", "As a recruiter, I want bulk resume parsing and tag assignment when uploading candidate archives, so that applicant screening requires less manual data entry.", "addition", "later", "customer", 2, 4, 4, "medium"),
    ("fr", "As a subscription user, I want self-service plan upgrades and prorated billing previews inside the billing portal, so that I do not need to contact sales.", "modification", "soon", "market", 3, 4, 6, "medium"),
    ("nfr", "As an accessibility specialist, I want WCAG 2.1 AA keyboard navigation and ARIA labels across all form controls, so that screen-reader users can complete workflows.", "modification", "soon", "organization", 1, 3, 6, "medium"),
    ("fr", "As a project coordinator, I want Gantt timeline dependencies visualized for linked backlog epics, so that milestone bottlenecks are spotted early.", "addition", "later", "customer", 2, 4, 4, "medium"),
    ("fr", "As a logistics operator, I want real-time carrier tracking status updates polled via REST webhooks, so that customers see accurate delivery windows.", "addition", "soon", "customer", 2, 3, 6, "medium"),
    ("nfr", "As a backend engineer, I want database connection pooling tuned with PgBouncer, so that serverless function bursts do not exhaust PostgreSQL max connections.", "modification", "soon", "developer knowledge", 2, 3, 5, "medium"),
    ("fr", "As an instructor, I want automated quiz grading with customizable rubric weights, so that students receive immediate feedback upon submission.", "addition", "soon", "customer", 2, 3, 6, "medium"),
    ("fr", "As a marketplace seller, I want promotional discount coupon codes with expiration dates and usage caps, so that I can run seasonal sales.", "addition", "later", "market", 2, 3, 6, "medium"),
    ("nfr", "As a QA lead, I want contract testing with Pact integrated between consumer and provider microservices, so that breaking API schema changes are caught in CI.", "addition", "later", "developer knowledge", 2, 4, 6, "medium"),
    ("fr", "As a fleet dispatcher, I want geofencing alerts triggered when delivery vehicles enter or exit designated warehouse zones, so that arrival times are logged.", "addition", "later", "customer", 3, 4, 4, "medium"),
    ("fr", "As a community moderator, I want automated spam keyword filtering and user report queues, so that abusive comments are hidden pending review.", "addition", "soon", "organization", 2, 3, 6, "medium"),
    ("nfr", "As a cloud FinOps analyst, I want automated tagging of Kubernetes namespaces and cloud volumes by cost center, so that monthly infrastructure spend is attributable.", "addition", "later", "organization", 1, 3, 5, "medium"),
    ("fr", "As a customer, I want saved search alerts emailed weekly when new listings match my filter criteria, so that I stay updated without daily visits.", "addition", "later", "market", 2, 3, 6, "medium"),
    ("fr", "As a finance user, I want multi-currency exchange rate conversion updated daily from an external FX feed, so that international invoices display local totals.", "addition", "soon", "customer", 2, 3, 6, "medium"),
    ("nfr", "As a release engineer, I want semantic versioning and automated changelog generation triggered on git tags, so that release notes remain consistent.", "addition", "later", "developer knowledge", 1, 4, 5, "medium"),
    ("fr", "As a clinic receptionist, I want SMS appointment reminders sent 24 hours prior to scheduled visits, so that patient no-show rates are reduced.", "addition", "soon", "customer", 2, 3, 5, "medium"),
    ("fr", "As a developer, I want GraphQL subscription support for live collaborative document cursors, so that multiple team members can co-edit specifications.", "addition", "later", "market", 3, 3, 4, "medium"),
    ("nfr", "As a security engineer, I want automated dependency update pull requests configured via Dependabot, so that minor library patches are applied regularly.", "addition", "later", "developer knowledge", 1, 4, 5, "medium"),

    # --- LOW PRIORITY (45 istanze moderne) ---
    ("fr", "As a user, I want a dark mode theme toggle persisted in local browser storage, so that the interface is more comfortable to read at night.", "addition", "later", "market", 0, 4, 5, "low"),
    ("fr", "As a profile owner, I want to upload a custom circular avatar image or choose from preset geometric icons, so that my account feels personalized.", "addition", "later", "market", 0, 4, 5, "low"),
    ("nfr", "As a UI designer, I want smooth CSS micro-animations on button hover and modal transitions, so that the web application feels polished.", "modification", "later", "project vision", 0, 4, 5, "low"),
    ("fr", "As a dashboard user, I want to rearrange widget cards via drag-and-drop and reset to default layout, so that I can customize my home screen.", "addition", "later", "customer", 1, 5, 6, "low"),
    ("fr", "As a reader, I want an estimated reading time badge displayed at the top of documentation articles, so that I know how long a guide takes.", "addition", "later", "organization", 0, 4, 5, "low"),
    ("fr", "As a user, I want keyboard shortcut tooltips shown when hovering over toolbar icons, so that I can learn hotkeys gradually.", "addition", "later", "developer knowledge", 0, 4, 5, "low"),
    ("nfr", "As a frontend developer, I want SVG icons consolidated into a single sprite sheet, so that minor asset requests are slightly cleaner.", "modification", "later", "developer knowledge", 0, 4, 5, "low"),
    ("fr", "As a member, I want to customize my email digest frequency between weekly and monthly summaries, so that my inbox stays tidy.", "modification", "later", "customer", 1, 5, 6, "low"),
    ("fr", "As a visitor, I want an interactive product tour modal on first login that can be dismissed permanently, so that new users see basic tips.", "addition", "later", "market", 1, 5, 6, "low"),
    ("fr", "As a user, I want custom color accent themes for workspace sidebars, so that I can visually distinguish staging from production environments.", "addition", "later", "market", 0, 4, 5, "low"),
    ("fr", "As an author, I want a word and character counter displayed below markdown comment boxes, so that I can track post length.", "addition", "later", "customer", 0, 4, 5, "low"),
    ("nfr", "As a designer, I want custom branded empty-state illustrations shown when a search query returns zero results, so that blank screens look friendly.", "addition", "later", "project vision", 0, 4, 5, "low"),
    ("fr", "As a user, I want to bookmark favorite reports into a quick-access sidebar folder, so that I save one click when navigating.", "addition", "later", "customer", 1, 5, 6, "low"),
    ("fr", "As a gamer, I want optional sound effects when completing daily checklist goals, so that task completion feels rewarding.", "addition", "later", "market", 0, 4, 5, "low"),
    ("fr", "As a portal user, I want localized date format preferences (DD/MM/YYYY vs MM/DD/YYYY) in account settings, so that dates match my region.", "addition", "later", "customer", 1, 5, 6, "low"),
    ("nfr", "As a frontend maintainer, I want unused legacy CSS classes pruned from the style bundle, so that stylesheet readability improves.", "modification", "later", "developer knowledge", 0, 4, 5, "low"),
    ("fr", "As a user, I want a 'Back to Top' floating button on long changelog pages, so that scrolling back up is convenient.", "addition", "later", "market", 0, 4, 5, "low"),
    ("fr", "As a workspace member, I want to set a custom emoji status message next to my display name, so that teammates know when I am at lunch.", "addition", "later", "market", 0, 4, 5, "low"),
    ("fr", "As a documentation reader, I want a one-click 'Copy Code' button on code blocks with a 'Copied!' toast, so that copying snippets is easier.", "addition", "later", "developer knowledge", 0, 4, 5, "low"),
    ("fr", "As a user, I want collapsible sidebar sections whose open/closed state is remembered across reloads, so that navigation stays compact.", "modification", "later", "customer", 0, 4, 5, "low"),
    ("fr", "As an admin, I want custom footer copyright text configurable from the branding settings page, so that white-label portals match company name.", "addition", "later", "customer", 0, 4, 5, "low"),
    ("nfr", "As a developer, I want Prettier formatting rules enforced in pre-commit hooks, so that code indentation remains uniform.", "addition", "later", "developer knowledge", 0, 4, 5, "low"),
    ("fr", "As a user, I want recently viewed items listed in a dropdown under the search bar, so that I can reopen recent pages.", "addition", "later", "market", 1, 5, 6, "low"),
    ("fr", "As a blog reader, I want social media share buttons for LinkedIn and X at the bottom of public release posts, so that I can share updates.", "addition", "later", "market", 0, 4, 5, "low"),
    ("fr", "As a user, I want pagination page-size selectors (10, 25, 50 rows) on secondary lookup tables, so that I can view more rows per page.", "modification", "later", "customer", 0, 4, 5, "low"),
    ("fr", "As a team member, I want celebratory confetti animation when all sprint tasks move to Done, so that team morale is boosted.", "addition", "later", "project vision", 0, 4, 5, "low"),
    ("fr", "As a user, I want compact vs comfortable table row density options in view settings, so that I can fit more data on small laptop screens.", "addition", "later", "market", 0, 4, 5, "low"),
    ("nfr", "As a UI engineer, I want custom scrollbar styling aligned with the brand color palette on Chromium browsers, so that scrollbars blend in.", "modification", "later", "project vision", 0, 4, 5, "low"),
    ("fr", "As a user, I want print-friendly CSS stylesheets for help center articles, so that printed pages hide navigation menus.", "addition", "later", "customer", 0, 4, 5, "low"),
    ("fr", "As a user, I want a release version number and build commit hash displayed in the help modal footer, so that support tickets can reference it.", "addition", "later", "developer knowledge", 0, 4, 5, "low"),
    ("fr", "As a customer, I want optional newsletter opt-in checkboxes during onboarding, so that I can receive product tips.", "addition", "later", "organization", 0, 4, 5, "low"),
    ("fr", "As a user, I want hover preview cards showing user bio and timezone when hovering over @mentions, so that context is visible inline.", "addition", "later", "market", 1, 5, 6, "low"),
    ("fr", "As a workspace user, I want customizable keyboard navigation shortcuts for archiving notifications, so that power users save time.", "addition", "later", "market", 1, 5, 6, "low"),
    ("fr", "As a portal visitor, I want an interactive FAQ accordion with smooth expand/collapse transitions, so that answers are easy to browse.", "addition", "later", "organization", 0, 4, 5, "low"),
    ("fr", "As a user, I want custom sound notification tones selectable in profile preferences, so that alerts are distinguishable.", "addition", "later", "market", 0, 4, 5, "low"),
    ("nfr", "As a developer, I want storybook documentation stories added for secondary badge components, so that UI kit examples are complete.", "addition", "later", "developer knowledge", 0, 4, 5, "low"),
    ("fr", "As a user, I want breadcrumb navigation links displayed above nested settings subpages, so that hierarchy is clear.", "addition", "later", "customer", 0, 4, 5, "low"),
    ("fr", "As a user, I want auto-expanding textarea height when typing long descriptions, so that internal scrollbars do not appear.", "modification", "later", "market", 0, 4, 5, "low"),
    ("fr", "As a dashboard viewer, I want a full-screen presentation mode button that hides top navigation bars, so that charts look clean on projectors.", "addition", "later", "customer", 0, 4, 5, "low"),
    ("fr", "As a user, I want optional weekly productivity streak badges on my profile card, so that consistent usage is gamified.", "addition", "later", "project vision", 0, 4, 5, "low"),
    ("fr", "As a user, I want a quick-filter pill to hide archived projects from the sidebar list, so that old workspaces stay out of sight.", "modification", "later", "customer", 0, 4, 5, "low"),
    ("fr", "As a user, I want tooltip explanations on secondary chart axes, so that novice users understand metric units.", "addition", "later", "organization", 0, 4, 5, "low"),
    ("fr", "As a user, I want customizable default landing page selection after login (Dashboard vs Backlog), so that I start on my preferred view.", "addition", "later", "customer", 0, 4, 5, "low"),
    ("fr", "As a user, I want a feedback emoji widget in the corner of beta features, so that quick sentiment ratings can be submitted.", "addition", "later", "organization", 0, 4, 5, "low"),
    ("fr", "As a user, I want high-contrast border toggle in theme preferences, so that card boundaries stand out on low-contrast monitors.", "addition", "later", "customer", 0, 4, 5, "low"),
]

df_runtime = pd.DataFrame(
    RUNTIME_STORIES,
    columns=["Type", "Requirement", "Change", "Urgency", "Origin", "N_dependencies", "Cost", "Time", "Priority"]
)
for pr in ["high", "medium", "low"]:
    m = df_runtime["Priority"] == pr
    sm = df_full[df_full["Priority"] == pr][["Change", "Urgency", "Origin", "N_dependencies", "Cost", "Time"]].sample(
        n=m.sum(), replace=True, random_state=SEED
    ).reset_index(drop=True)
    df_runtime.loc[m, sm.columns] = sm.values
df_runtime.to_csv("dataset/runtime_benchmark.csv", index=False)

# ==============================================================================
# 4. EMBEDDING E ANALISI COVARIATE SHIFT / OOV (Sez. 3.1.2 & 4.1.2)
# ==============================================================================
print("\n[1/6] Calcolo Dense Sentence Embeddings (all-MiniLM-L6-v2) e analisi OOV...")
sbert = SentenceTransformer("models/all-MiniLM-L6-v2", device=device.type)

def get_norm_embeddings(texts):
    emb = sbert.encode(list(texts), show_progress_bar=False, device=device.type)
    return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)

train_base["clean_req"] = train_base["Requirement"].apply(clean_requirement_text)
test_lab["clean_req"] = test_lab["Requirement"].apply(clean_requirement_text)
df_runtime["clean_req"] = df_runtime["Requirement"].apply(clean_requirement_text)

tfidf_check = TfidfVectorizer()
tfidf_check.fit(train_base["Requirement"].str.lower())
vocab_hist = set(tfidf_check.vocabulary_.keys())
runtime_tokens = [tok for req in df_runtime["Requirement"].str.lower() for tok in re.findall(r"\b[a-z0-9_]+\b", req)]
oov_count = sum(1 for t in runtime_tokens if t not in vocab_hist)
oov_rate = round(100 * oov_count / max(len(runtime_tokens), 1), 2)
len_promise = round(train_base["Requirement"].str.split().str.len().mean(), 1)
len_runtime = round(df_runtime["Requirement"].str.split().str.len().mean(), 1)
print(f"Tasso OOV TF-IDF a runtime: {oov_rate}% | Lunghezza media token: PROMISE={len_promise}, Runtime={len_runtime}")

# ==============================================================================
# 5. OFFLINE DATA AUGMENTATION VIA GROQ LLM + FILTRO METRICO (Sez. 3.1.3)
# ==============================================================================
print("\n[2/6] Offline Data Augmentation via Groq LLM & Filtraggio Coseno (tau_sim, tau_dup)...")
AUG_CACHE_PATH = "dataset/groq_augmented_train.csv"
TAU_SIM = 0.48
TAU_DUP = 0.90

def call_groq_json(prompt_text: str):
    if not GROQ_API_KEY:
        return None
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    for model_name in ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"]:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a Senior Requirements Engineer. Return ONLY valid JSON."},
                {"role": "user", "content": prompt_text}
            ],
            "temperature": 0.5,
            "top_p": 0.9,
            "response_format": {"type": "json_object"}
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=45)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                m = re.search(r"\{[\s\S]*\}", content)
                if m:
                    return json.loads(m.group(0))
        except Exception:
            continue
    return None

if os.path.exists(AUG_CACHE_PATH):
    df_aug_only = pd.read_csv(AUG_CACHE_PATH)
    aug_stats = json.load(open("dataset/groq_aug_stats.json"))
    print(f"Caricata cache Groq Augmentation: {len(df_aug_only)} istanze accettate.")
else:
    trans_sample = train_base.sample(n=100, random_state=SEED).reset_index(drop=True)
    candidates = []
    n_trans_in = len(trans_sample)
    n_gen_in = 90

    # 1. Domain Translation (batch da 10)
    for start in range(0, n_trans_in, 10):
        batch = trans_sample.iloc[start:start+10]
        items_str = "\n".join([f"{idx}: [{r['Priority'].upper()}] {r['Requirement']}" for idx, r in batch.iterrows()])
        prompt = f"""Translate these legacy software requirements into modern Agile User Stories ("As a [role], I want [capability], so that [business value]") using contemporary cloud/API/security terminology while preserving their exact functional meaning and priority level.
Return JSON: {{"items": [{{"id": 0, "cot_reasoning": "...", "agile_story": "..."}}]}}
Requirements:
{items_str}"""
        res = call_groq_json(prompt)
        if res and "items" in res:
            for item in res["items"]:
                idx = int(item.get("id", -1))
                if idx in batch.index and item.get("agile_story"):
                    orig = batch.loc[idx]
                    candidates.append({
                        "Type": orig["Type"],
                        "Requirement": item["agile_story"].strip(),
                        "Change": orig["Change"],
                        "Urgency": orig["Urgency"],
                        "Origin": orig["Origin"],
                        "N_dependencies": orig["N_dependencies"],
                        "Cost": orig["Cost"],
                        "Time": orig["Time"],
                        "Priority": orig["Priority"],
                        "source": "translation",
                        "orig_req": orig["Requirement"]
                    })
        time.sleep(0.4)

    # 2. Targeted Minority Class Generation (High Priority) con Chain-of-Thought
    scenarios = [
        "OAuth2/JWT API authentication and zero-trust session security",
        "Kubernetes high availability, multi-AZ failover and disaster recovery",
        "GDPR data privacy, PII encryption at rest and immutable audit logging",
        "Distributed ACID transactions, payment idempotency and fraud prevention",
        "API Gateway rate limiting, circuit breakers and DDoS resilience",
        "CI/CD container CVE vulnerability blocking and supply-chain security",
        "Low-latency SLA enforcement (<50ms p99) under high concurrency",
        "Database row-level multi-tenant isolation and SQL/GraphQL injection prevention",
        "OpenTelemetry real-time incident alerting and SLA error-budget protection"
    ]
    for scenario in scenarios:
        prompt = f"""Generate 10 distinct, realistic HIGH-priority Agile User Stories ("As a [role], I want [capability], so that [business value]") for modern software systems focusing on: {scenario}.
Use strong deontic constraints (must/shall) and realistic engineering details.
Return JSON: {{"items": [{{"cot_reasoning": "...", "type": "nfr", "change": "addition", "urgency": "soon", "origin": "organization", "n_dependencies": 2, "cost": 2, "time": 3, "agile_story": "..."}}]}}"""
        res = call_groq_json(prompt)
        if res and "items" in res:
            for item in res["items"]:
                if item.get("agile_story"):
                    candidates.append({
                        "Type": item.get("type", "nfr").lower() if item.get("type", "nfr").lower() in ("fr", "nfr") else "nfr",
                        "Requirement": item["agile_story"].strip(),
                        "Change": "addition",
                        "Urgency": "soon",
                        "Origin": item.get("origin", "organization").lower() if item.get("origin", "organization").lower() in ("organization", "market", "customer", "developer knowledge", "project vision") else "organization",
                        "N_dependencies": int(np.clip(item.get("n_dependencies", 2), 1, 3)),
                        "Cost": int(np.clip(item.get("cost", 2), 2, 3)),
                        "Time": int(np.clip(item.get("time", 3), 2, 4)),
                        "Priority": "high",
                        "source": "minority_gen",
                        "orig_req": ""
                    })
        time.sleep(0.4)

    train_emb_base = get_norm_embeddings(train_base["Requirement"].apply(clean_requirement_text))
    accepted_rows = []
    accepted_embs = []
    scarti_sim = 0
    scarti_dup = 0
    tot_trans_eval = 0

    for cand in candidates:
        cand_clean = clean_requirement_text(cand["Requirement"])
        cand_emb = get_norm_embeddings([cand_clean])[0]
        if cand["source"] == "translation":
            tot_trans_eval += 1
            orig_emb = get_norm_embeddings([clean_requirement_text(cand["orig_req"])])[0]
            sim_orig = float(np.dot(cand_emb, orig_emb))
            if sim_orig < TAU_SIM:
                scarti_sim += 1
                continue
        pool_embs = train_emb_base if len(accepted_embs) == 0 else np.vstack([train_emb_base, np.array(accepted_embs)])
        max_dup = float(np.max(pool_embs @ cand_emb))
        if max_dup > TAU_DUP:
            scarti_dup += 1
            continue
        accepted_embs.append(cand_emb)
        accepted_rows.append({k: v for k, v in cand.items() if k not in ("source", "orig_req")})

    df_aug_only = pd.DataFrame(accepted_rows)
    df_aug_only.to_csv(AUG_CACHE_PATH, index=False)
    tot_cand = max(len(candidates), 1)
    aug_stats = {
        "n_trans_in": n_trans_in,
        "n_gen_in": n_gen_in,
        "tau_sim": TAU_SIM,
        "tau_dup": TAU_DUP,
        "scarti_sim_pct": round(100 * scarti_sim / max(tot_trans_eval, 1), 1),
        "scarti_dup_pct": round(100 * scarti_dup / tot_cand, 1),
        "n_acc_llm": len(df_aug_only),
        "acc_netto_pct": round(100 * len(df_aug_only) / tot_cand, 1)
    }
    json.dump(aug_stats, open("dataset/groq_aug_stats.json", "w"), indent=2)
    print(f"Nuove istanze Groq accettate: {len(df_aug_only)} (su {tot_cand} candidate).")

train_aug = pd.concat([train_base.drop(columns=["clean_req"]), df_aug_only], ignore_index=True)
train_aug["clean_req"] = train_aug["Requirement"].apply(clean_requirement_text)

# ==============================================================================
# 6. VALUTAZIONE GRAPH NEURAL NETWORK (Sez. 3.2.3 & 4.3.1 - Tabella 4.5)
# ==============================================================================
print("\n[3/6] Addestramento e valutazione GNN induttiva vs Baseline (Link Prediction)...")

class InductiveGNNLinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, p_drop=0.35, p_edge=0.20):
        super().__init__()
        self.p_edge = p_edge
        self.p_drop = p_drop
        self.lin_in = nn.Linear(in_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_channels, 1)
        )

    def encode_nodes(self, x, edge_index):
        if self.training and self.p_edge > 0 and edge_index.size(1) > 0:
            mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.p_edge
            edge_index = edge_index[:, mask]
        h0 = F.relu(self.norm1(self.lin_in(x)))
        h1 = F.relu(self.norm2(self.conv1(h0, edge_index) + h0))
        h1 = F.dropout(h1, p=self.p_drop, training=self.training)
        h2 = F.relu(self.conv2(h1, edge_index) + h1)
        return h2

    def decode_edges(self, z, edge_pairs):
        zu, zv = z[edge_pairs[0]], z[edge_pairs[1]]
        feat = torch.cat([zu, zv, zu * zv], dim=-1)
        return self.edge_mlp(feat).squeeze(-1)

def build_ground_truth_graph(df_sub: pd.DataFrame, emb_mat: np.ndarray, sim_th=0.58):
    sim = cosine_similarity(emb_mat)
    n = len(df_sub)
    pos_edges = []
    neg_edges = []
    for i in range(n):
        target_deg = int(df_sub.loc[i, "N_dependencies"])
        same_origin = (df_sub["Origin"].values == df_sub.loc[i, "Origin"]) & (df_sub["Type"].values == df_sub.loc[i, "Type"])
        scores = sim[i] + 0.12 * same_origin.astype(float)
        scores[i] = -1.0
        top_k = max(target_deg, 1) if target_deg > 0 else 0
        top_neighbors = set(np.argsort(scores)[::-1][:top_k]) if top_k > 0 else set()
        for j in range(i + 1, n):
            if j in top_neighbors or sim[i, j] >= sim_th:
                pos_edges.append((i, j))
            else:
                neg_edges.append((i, j))
    return pos_edges, neg_edges

cat_encoder_gnn = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
cat_train_gnn = cat_encoder_gnn.fit_transform(train_aug[["Type", "Change", "Urgency", "Origin"]])
cat_rt_gnn = cat_encoder_gnn.transform(df_runtime[["Type", "Change", "Urgency", "Origin"]])

emb_train_aug = get_norm_embeddings(train_aug["clean_req"])
emb_rt_clean = get_norm_embeddings(df_runtime["clean_req"])

pos_tr, neg_tr = build_ground_truth_graph(train_aug, emb_train_aug, sim_th=0.42)
pos_rt, neg_rt = build_ground_truth_graph(df_runtime, emb_rt_clean, sim_th=0.38)

tfidf_gnn = TfidfVectorizer(max_features=384)
x_tr_tfidf = np.hstack([tfidf_gnn.fit_transform(train_aug["Requirement"]).toarray(), cat_train_gnn])
x_rt_tfidf = np.hstack([tfidf_gnn.transform(df_runtime["Requirement"]).toarray(), cat_rt_gnn])

def train_and_eval_gnn(x_tr_np, x_rt_np, weighted=True, p_edge=0.20):
    x_tr = torch.tensor(x_tr_np, dtype=torch.float, device=device)
    x_rt = torch.tensor(x_rt_np, dtype=torch.float, device=device)
    edge_index_tr = torch.tensor(pos_tr + [(v, u) for u, v in pos_tr], dtype=torch.long, device=device).t().contiguous()
    known_rt = pos_rt[:max(1, len(pos_rt)//5)]
    edge_index_rt = torch.tensor(known_rt + [(v, u) for u, v in known_rt], dtype=torch.long, device=device).t().contiguous()

    gnn = InductiveGNNLinkPredictor(x_tr.shape[1], hidden_channels=64, p_drop=0.35 if weighted else 0.0, p_edge=p_edge).to(device)
    opt = torch.optim.AdamW(gnn.parameters(), lr=1e-3, weight_decay=1e-3)
    w_pos = torch.tensor([3.5 if weighted else 1.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=w_pos)

    pos_t = torch.tensor(pos_tr, dtype=torch.long, device=device).t()
    rng_gnn = np.random.default_rng(SEED)

    for _ in range(45):
        gnn.train()
        opt.zero_grad()
        neg_idx = rng_gnn.choice(len(neg_tr), size=min(len(neg_tr), len(pos_tr) * 3), replace=False)
        neg_t = torch.tensor([neg_tr[i] for i in neg_idx], dtype=torch.long, device=device).t()
        pairs = torch.cat([pos_t, neg_t], dim=1)
        labels = torch.cat([torch.ones(pos_t.size(1), device=device), torch.zeros(neg_t.size(1), device=device)])
        z = gnn.encode_nodes(x_tr, edge_index_tr)
        logits = gnn.decode_edges(z, pairs)
        loss = criterion(logits, labels)
        loss.backward()
        opt.step()

    gnn.eval()
    with torch.no_grad():
        z_rt = gnn.encode_nodes(x_rt, edge_index_rt)
        eval_pairs = pos_rt + neg_rt
        eval_t = torch.tensor(eval_pairs, dtype=torch.long, device=device).t()
        probs = torch.sigmoid(gnn.decode_edges(z_rt, eval_t)).cpu().numpy()
        y_true = np.array([1]*len(pos_rt) + [0]*len(neg_rt))
        preds = (probs >= 0.5).astype(int)
        return {
            "precision": round(100 * precision_score(y_true, preds, zero_division=0), 2),
            "recall": round(100 * recall_score(y_true, preds, zero_division=0), 2),
            "f1": round(100 * f1_score(y_true, preds, zero_division=0), 2),
            "pr_auc": round(float(average_precision_score(y_true, probs)), 3)
        }, gnn

gnn_base_res, _ = train_and_eval_gnn(x_tr_tfidf, x_rt_tfidf, weighted=False, p_edge=0.0)
x_tr_dense = np.hstack([emb_train_aug, cat_train_gnn])
x_rt_dense = np.hstack([emb_rt_clean, cat_rt_gnn])
gnn_opt_res, gnn_opt_model = train_and_eval_gnn(x_tr_dense, x_rt_dense, weighted=True, p_edge=0.20)
print("GNN Baseline:", gnn_base_res)
print("GNN Ottimizzata:", gnn_opt_res)

def extract_struct_features(df_sub: pd.DataFrame, emb_mat: np.ndarray) -> np.ndarray:
    sim = cosine_similarity(emb_mat)
    np.fill_diagonal(sim, 0.0)
    deg_centrality = (sim > 0.55).sum(axis=1) / max(len(df_sub) - 1, 1)
    max_sim = sim.max(axis=1)
    mean_top3 = np.sort(sim, axis=1)[:, -3:].mean(axis=1)
    req_len = df_sub["Requirement"].str.split().str.len().values
    has_deontic = df_sub["Requirement"].str.lower().str.contains(r"\b(must|shall|never|cannot)\b").astype(float).values
    num_cols = df_sub[["N_dependencies", "Cost", "Time"]].astype(float).values
    return np.column_stack([num_cols, deg_centrality, max_sim, mean_top3, req_len, has_deontic])

# ==============================================================================
# 7. ABLATION STUDY: CONFIG A, B, C, D (Sez. 3.3.2 & 4.2 - Tabelle 4.1, 4.3, 4.4)
# ==============================================================================
print("\n[4/6] Esecuzione Ablation Study (Config. A -> B -> C -> D)...")

target_enc = OrdinalEncoder(categories=[["low", "medium", "high"]])
y_train_base = target_enc.fit_transform(train_base[["Priority"]]).ravel().astype(int)
y_test_lab = target_enc.transform(test_lab[["Priority"]]).ravel().astype(int)
y_train_aug = target_enc.transform(train_aug[["Priority"]]).ravel().astype(int)
y_runtime = target_enc.transform(df_runtime[["Priority"]]).ravel().astype(int)

cat_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
cat_tr_base = cat_enc.fit_transform(train_base[["Type", "Change", "Urgency", "Origin"]])
cat_te_lab = cat_enc.transform(test_lab[["Type", "Change", "Urgency", "Origin"]])
cat_tr_aug = cat_enc.transform(train_aug[["Type", "Change", "Urgency", "Origin"]])
cat_rt = cat_enc.transform(df_runtime[["Type", "Change", "Urgency", "Origin"]])

emb_tr_base = get_norm_embeddings(train_base["clean_req"])
emb_te_lab = get_norm_embeddings(test_lab["clean_req"])

struct_tr_base = extract_struct_features(train_base, emb_tr_base)
struct_te_lab = extract_struct_features(test_lab, emb_te_lab)
struct_tr_aug = extract_struct_features(train_aug, emb_train_aug)
struct_rt = extract_struct_features(df_runtime, emb_rt_clean)

def evaluate_predictions(y_true, y_pred, y_proba):
    acc = 100 * accuracy_score(y_true, y_pred)
    bal_acc = 100 * balanced_accuracy_score(y_true, y_pred)
    macro_f1 = 100 * f1_score(y_true, y_pred, average="macro", zero_division=0)
    rec_high = 100 * recall_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0]
    prec_high = 100 * precision_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0]
    f1_high = 100 * f1_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0]
    y_true_high = (y_true == 2).astype(int)
    pr_auc_high = average_precision_score(y_true_high, y_proba[:, 2])
    y_onehot = np.eye(3)[y_true]
    bs = float(np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1)))
    high_mask = (y_true == 2)
    inv_high_low = 100 * np.mean(y_pred[high_mask] == 0) if high_mask.sum() > 0 else 0.0
    return {
        "acc": round(acc, 2),
        "bal_acc": round(bal_acc, 2),
        "macro_f1": round(macro_f1, 2),
        "rec_high": round(rec_high, 2),
        "prec_high": round(prec_high, 2),
        "f1_high": round(f1_high, 2),
        "pr_auc_high": round(float(pr_auc_high), 3),
        "brier": round(bs, 3),
        "inv_high_low": round(float(inv_high_low), 2)
    }

# --- CONFIG A: Baseline Originaria (TF-IDF + RF Standard) ---
tfidf_a = TfidfVectorizer(max_features=300)
X_tr_a = np.hstack([tfidf_a.fit_transform(train_base["Requirement"]).toarray(), cat_tr_base, train_base[["N_dependencies", "Cost", "Time"]].values])
X_lab_a = np.hstack([tfidf_a.transform(test_lab["Requirement"]).toarray(), cat_te_lab, test_lab[["N_dependencies", "Cost", "Time"]].values])
X_rt_a = np.hstack([tfidf_a.transform(df_runtime["Requirement"]).toarray(), cat_rt, df_runtime[["N_dependencies", "Cost", "Time"]].values])

rf_a = RandomForestClassifier(n_estimators=100, random_state=SEED)
rf_a.fit(X_tr_a, y_train_base)
res_a_lab = evaluate_predictions(y_test_lab, rf_a.predict(X_lab_a), rf_a.predict_proba(X_lab_a))
res_a_rt = evaluate_predictions(y_runtime, rf_a.predict(X_rt_a), rf_a.predict_proba(X_rt_a))

# --- CONFIG B: Pulizia spaCy + Dense Embeddings all-MiniLM-L6-v2 (senza resampling) ---
X_tr_b = np.hstack([emb_tr_base, cat_tr_base, struct_tr_base])
X_lab_b = np.hstack([emb_te_lab, cat_te_lab, struct_te_lab])
X_rt_b = np.hstack([emb_rt_clean, cat_rt, struct_rt])

rf_b = RandomForestClassifier(n_estimators=100, random_state=SEED)
rf_b.fit(X_tr_b, y_train_base)
res_b_lab = evaluate_predictions(y_test_lab, rf_b.predict(X_lab_b), rf_b.predict_proba(X_lab_b))
res_b_rt = evaluate_predictions(y_runtime, rf_b.predict(X_rt_b), rf_b.predict_proba(X_rt_b))

# --- HYPERPARAMETER TUNING & CONFIG C / D ---
def build_pipeline(k_pca=32, k_nn=5, n_est=200, max_depth=10, min_leaf=3, sampler_type="borderline"):
    ct = ColumnTransformer([
        ("sem", ImbPipeline([("pca", PCA(n_components=k_pca, random_state=SEED)), ("std", StandardScaler())]), list(range(384))),
        ("struct", RobustScaler(), list(range(384, 384 + 4 + 8)))
    ])
    if sampler_type == "smote":
        sampler = SMOTE(k_neighbors=k_nn, random_state=SEED)
    elif sampler_type == "tomek":
        sampler = SMOTETomek(smote=SMOTE(k_neighbors=k_nn, random_state=SEED), random_state=SEED)
    else:
        sampler = BorderlineSMOTE(k_neighbors=k_nn, random_state=SEED)

    rf = RandomForestClassifier(
        n_estimators=n_est, max_depth=max_depth, min_samples_leaf=min_leaf,
        class_weight="balanced_subsample", random_state=SEED, n_jobs=-1
    )
    return ImbPipeline([("prep", ct), ("res", sampler), ("clf", rf)])

X_tr_d = np.hstack([emb_train_aug, cat_tr_aug, struct_tr_aug])
X_rt_d = np.hstack([emb_rt_clean, cat_rt, struct_rt])

param_dist = {
    "prep__sem__pca__n_components": [16, 24, 32, 48],
    "res__k_neighbors": [3, 5, 7],
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [8, 10, 12, 15],
    "clf__min_samples_leaf": [2, 3, 4, 5]
}
base_search_pipe = build_pipeline()
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
search = RandomizedSearchCV(
    base_search_pipe, param_distributions=param_dist, n_iter=25,
    scoring="f1_macro", cv=cv5, random_state=SEED, n_jobs=-1
)
search.fit(X_tr_d, y_train_aug)
best_params = search.best_params_
print("Migliori Iperparametri trovati:", best_params)

k_opt = best_params["prep__sem__pca__n_components"]
k_nn_opt = best_params["res__k_neighbors"]
n_est_opt = best_params["clf__n_estimators"]
depth_opt = best_params["clf__max_depth"]
leaf_opt = best_params["clf__min_samples_leaf"]

def fit_calibrated_and_tune_threshold(X_train_mat, y_train_vec):
    pipe_uncal = build_pipeline(k_opt, k_nn_opt, n_est_opt, depth_opt, leaf_opt, "borderline")
    pipe_uncal.fit(X_train_mat, y_train_vec)

    cal_sig = CalibratedClassifierCV(
        build_pipeline(k_opt, k_nn_opt, n_est_opt, depth_opt, leaf_opt, "borderline"),
        method="sigmoid", cv=5
    )
    cal_sig.fit(X_train_mat, y_train_vec)

    cal_iso = CalibratedClassifierCV(
        build_pipeline(k_opt, k_nn_opt, n_est_opt, depth_opt, leaf_opt, "borderline"),
        method="isotonic", cv=5
    )
    cal_iso.fit(X_train_mat, y_train_vec)

    oof_probs = np.zeros((len(y_train_vec), 3))
    for tr_i, va_i in cv5.split(X_train_mat, y_train_vec):
        m_fold = CalibratedClassifierCV(
            build_pipeline(k_opt, k_nn_opt, n_est_opt, depth_opt, leaf_opt, "borderline"),
            method="sigmoid", cv=3
        )
        m_fold.fit(X_train_mat[tr_i], y_train_vec[tr_i])
        oof_probs[va_i] = m_fold.predict_proba(X_train_mat[va_i])

    best_th, best_fb = 0.40, -1.0
    beta = 1.1
    y_true_h = (y_train_vec == 2).astype(int)
    for th in np.linspace(0.30, 0.65, 36):
        pred_h = (oof_probs[:, 2] >= th).astype(int)
        p = precision_score(y_true_h, pred_h, zero_division=0)
        r = recall_score(y_true_h, pred_h, zero_division=0)
        if p + r > 0:
            fb = (1 + beta**2) * p * r / (beta**2 * p + r)
            if fb > best_fb:
                best_fb, best_th = fb, float(th)

    return pipe_uncal, cal_sig, cal_iso, round(best_th, 2)

def predict_with_threshold(probs, theta_high):
    preds = np.argmax(probs, axis=1)
    preds[probs[:, 2] >= theta_high] = 2
    return preds

# Config C (su train_base senza LLM Augmentation)
uncal_c, sig_c, iso_c, th_c = fit_calibrated_and_tune_threshold(X_tr_b, y_train_base)
res_c_lab = evaluate_predictions(y_test_lab, predict_with_threshold(sig_c.predict_proba(X_lab_b), th_c), sig_c.predict_proba(X_lab_b))
res_c_rt_uncal = evaluate_predictions(y_runtime, uncal_c.predict(X_rt_b), uncal_c.predict_proba(X_rt_b))
res_c_rt = evaluate_predictions(y_runtime, predict_with_threshold(sig_c.predict_proba(X_rt_b), th_c), sig_c.predict_proba(X_rt_b))

# Config D (Full Pipeline: train_aug + Calibrazione + Soglia ottima)
uncal_d, sig_d, iso_d, th_d = fit_calibrated_and_tune_threshold(X_tr_d, y_train_aug)
res_d_lab = evaluate_predictions(y_test_lab, predict_with_threshold(sig_d.predict_proba(X_lab_b), th_d), sig_d.predict_proba(X_lab_b))
res_d_rt_uncal = evaluate_predictions(y_runtime, uncal_d.predict(X_rt_d), uncal_d.predict_proba(X_rt_d))
res_d_rt_sig = evaluate_predictions(y_runtime, predict_with_threshold(sig_d.predict_proba(X_rt_d), th_d), sig_d.predict_proba(X_rt_d))
res_d_rt_iso = evaluate_predictions(y_runtime, predict_with_threshold(iso_d.predict_proba(X_rt_d), th_d), iso_d.predict_proba(X_rt_d))

first_tr_idx, _ = next(cv5.split(X_tr_d, y_train_aug))
_, y_fold_res = BorderlineSMOTE(k_neighbors=k_nn_opt, random_state=SEED).fit_resample(X_tr_d[first_tr_idx], y_train_aug[first_tr_idx])
fold_res_dist = get_dist_stats(pd.DataFrame({"Priority": target_enc.inverse_transform(y_fold_res.reshape(-1, 1)).ravel()}))

# ==============================================================================
# 8. PERMUTATION IMPORTANCE (Sez. 4.3.2)
# ==============================================================================
print("\n[5/6] Calcolo Permutation Importance (Dimensionality Dominance)...")
perm_raw = permutation_importance(rf_b, X_rt_b, y_runtime, n_repeats=10, scoring="f1_macro", random_state=SEED, n_jobs=-1)
imp_raw = np.maximum(perm_raw.importances_mean, 0)
tot_raw = max(imp_raw.sum(), 1e-9)
imp_struct_raw_pct = round(100 * imp_raw[384:].sum() / tot_raw, 1)

perm_opt = permutation_importance(sig_d, X_rt_d, y_runtime, n_repeats=10, scoring="f1_macro", random_state=SEED, n_jobs=-1)
imp_opt = np.maximum(perm_opt.importances_mean, 0)
tot_opt = max(imp_opt.sum(), 1e-9)
imp_sem_opt_pct = round(100 * imp_opt[:384].sum() / tot_opt, 1)
imp_struct_opt_pct = round(100 * imp_opt[384:].sum() / tot_opt, 1)

# ==============================================================================
# 9. SIMULAZIONE HUMAN-IN-THE-LOOP & CONCEPT DRIFT SU 6 SPRINT (Sez. 4.4.1)
# ==============================================================================
# ==============================================================================
# 9. SIMULAZIONE HUMAN-IN-THE-LOOP & CONCEPT DRIFT SU 6 SPRINT (Sez. 4.4.1)
# ==============================================================================
print("\n[6/6] Simulazione longitudinale Concept Drift (6 Sprint) e Latenza...")
perm_idx = np.random.default_rng(SEED).permutation(len(df_runtime))
df_stream = df_runtime.iloc[perm_idx].reset_index(drop=True)
X_rt_stream = X_rt_d[perm_idx]
y_stream_base = y_runtime[perm_idx]

drift_mask = df_stream["Requirement"].str.lower().str.contains("logging|gdpr|analytics|compliance|audit|webhook|caching|tracking|security|api")
y_stream_pre = y_stream_base.copy()
y_stream_post = y_stream_base.copy()
y_stream_post[drift_mask & (y_stream_post == 1)] = 2

sprints = np.array_split(np.arange(len(df_stream)), 6)

# Strategia 1: Modello Statico (Config D)
static_f1s = []
for s_idx, idx_s in enumerate(sprints):
    y_true_s = y_stream_pre[idx_s] if s_idx < 2 else y_stream_post[idx_s]
    p_s = predict_with_threshold(sig_d.predict_proba(X_rt_stream[idx_s]), th_d)
    static_f1s.append(100 * f1_score(y_true_s, p_s, average="macro", zero_division=0))
static_rec_end = round(100 * recall_score(
    y_stream_post[np.concatenate(sprints[4:])],
    predict_with_threshold(sig_d.predict_proba(X_rt_stream[np.concatenate(sprints[4:])]), th_d),
    labels=[2], average=None
)[0], 2)

# Strategia 2 & 3: Incremental Non Pesato vs HITL Pesato + Entropia + Validation Gate
X_buf_u, y_buf_u = X_tr_d.copy(), y_train_aug.copy()
X_buf_h, y_buf_h, w_buf_h = X_tr_d.copy(), y_train_aug.copy(), np.ones(len(y_train_aug))
unweighted_f1s, hitl_f1s = [], []
gate_u, gate_h = 0, 0
ALPHA_OP = 12.0
LAMBDA_DECAY = 0.15

model_u = sig_d
model_h = sig_d

for s_idx, idx_s in enumerate(sprints):
    y_true_s = y_stream_pre[idx_s] if s_idx < 2 else y_stream_post[idx_s]
    probs_u = model_u.predict_proba(X_rt_stream[idx_s])
    probs_h = model_h.predict_proba(X_rt_stream[idx_s])

    unweighted_f1s.append(100 * f1_score(y_true_s, predict_with_threshold(probs_u, th_d), average="macro", zero_division=0))
    hitl_f1s.append(100 * f1_score(y_true_s, predict_with_threshold(probs_h, th_d), average="macro", zero_division=0))

    ent = -np.sum(probs_h * np.log2(np.clip(probs_h, 1e-9, 1.0)), axis=1)
    k_sel = max(2, int(0.20 * len(idx_s)))
    sel_local = np.argsort(ent)[::-1][:k_sel]
    sel_global = idx_s[sel_local]

    # Aggiornamento Unweighted
    X_buf_u = np.vstack([X_buf_u, X_rt_stream[sel_global]])
    y_buf_u = np.concatenate([y_buf_u, y_true_s[sel_local]])
    cand_u = build_pipeline(k_opt, k_nn_opt, n_est_opt, depth_opt, leaf_opt, "borderline")
    cand_u.fit(X_buf_u, y_buf_u)
    model_u = cand_u
    gate_u += 1

    # Aggiornamento HITL Pesato + Validation Gate
    w_buf_h = np.maximum(0.5, w_buf_h * np.exp(-LAMBDA_DECAY))
    X_buf_h = np.vstack([X_buf_h, X_rt_stream[sel_global]])
    y_buf_h = np.concatenate([y_buf_h, y_true_s[sel_local]])
    w_buf_h = np.concatenate([w_buf_h, np.full(len(sel_global), ALPHA_OP)])

    cand_h = build_pipeline(k_opt, k_nn_opt, n_est_opt, depth_opt, leaf_opt, "borderline")
    X_prep_h = cand_h.named_steps["prep"].fit_transform(X_buf_h, y_buf_h)
    X_res_h, y_res_h = cand_h.named_steps["res"].fit_resample(X_prep_h, y_buf_h)
    w_res_h = np.concatenate([w_buf_h, np.ones(len(y_res_h) - len(y_buf_h))])
    cand_h.named_steps["clf"].fit(X_res_h, y_res_h, sample_weight=w_res_h)

    if balanced_accuracy_score(y_test_lab, cand_h.predict(X_lab_b)) >= (res_d_lab["bal_acc"] / 100.0 - 0.03):
        model_h = cand_h
        gate_h += 1

idx_end = np.concatenate(sprints[4:])
y_end = y_stream_post[idx_end]
rec_u_end = round(100 * recall_score(y_end, predict_with_threshold(model_u.predict_proba(X_rt_stream[idx_end]), th_d), labels=[2], average=None)[0], 2)
rec_h_end = round(100 * recall_score(y_end, predict_with_threshold(model_h.predict_proba(X_rt_stream[idx_end]), th_d), labels=[2], average=None)[0], 2)

# Benchmark Latenza locale su 15 User Story (Sez. 4.4.2)
t0 = time.perf_counter()
demo_clean = [clean_requirement_text(r) for r in df_runtime["Requirement"].iloc[:15]]
demo_emb = get_norm_embeddings(demo_clean)
demo_struct = extract_struct_features(df_runtime.iloc[:15].reset_index(drop=True), demo_emb)
X_demo = np.hstack([demo_emb, cat_rt[:15], demo_struct])
demo_probs = sig_d.predict_proba(X_demo)
demo_ent = -np.sum(demo_probs * np.log2(np.clip(demo_probs, 1e-9, 1.0)), axis=1)
lat_locale_ms = round((time.perf_counter() - t0) * 1000, 1)
n_alert_demo = int(np.sum(demo_ent > 1.15))

# ==============================================================================
# 10. SALVATAGGIO MODELLI OTTIMIZZATI E REPORT FINALE
# ==============================================================================
joblib.dump(sig_d, "models/classifier_optimized.pkl")
joblib.dump(target_enc, "models/target_encoder_optimized.pkl")
torch.save(gnn_opt_model.state_dict(), "models/gnn_link_optimized.pt")
with open("models/opt_config.json", "w") as f:
    json.dump({"theta_high": th_d, "k_pca": int(k_opt)}, f, indent=2)

report = {
    "hardware_4_1_1": hw_specs,
    "hyperparams_table_4_1": {
        "k_opt": int(k_opt), "algo_opt": "BorderlineSMOTE", "k_nn": int(k_nn_opt),
        "n_est_opt": int(n_est_opt), "depth_opt": int(depth_opt), "leaf_opt": int(leaf_opt),
        "metodo_cal": "sigmoid / isotonic", "theta_opt": th_d,
        "gnn_arch": "GraphSAGE", "dh_opt": 64, "K_opt": 2, "p_edge": 0.20, "p_drop": 0.35,
        "lr_opt": 0.001, "wd_opt": 0.001, "w_pos": 3.5
    },
    "datasets_table_4_2": {
        "promise_originario": get_dist_stats(df_full),
        "lab_test_set": get_dist_stats(test_lab),
        "training_base": get_dist_stats(train_base),
        "training_plus_groq": get_dist_stats(train_aug),
        "singolo_fold_post_resampling": fold_res_dist,
        "runtime_benchmark": get_dist_stats(df_runtime),
        "augmentation_stats": aug_stats,
        "oov_rate_pct": oov_rate,
        "len_promise": len_promise,
        "len_runtime": len_runtime
    },
    "ablation_table_4_3": {
        "Config_A_Lab": res_a_lab, "Config_A_Runtime": res_a_rt, "Gap_A": round(res_a_lab["bal_acc"] - res_a_rt["bal_acc"], 2),
        "Config_B_Lab": res_b_lab, "Config_B_Runtime": res_b_rt, "Gap_B": round(res_b_lab["bal_acc"] - res_b_rt["bal_acc"], 2),
        "Config_C_Lab": res_c_lab, "Config_C_Runtime": res_c_rt, "Gap_C": round(res_c_lab["bal_acc"] - res_c_rt["bal_acc"], 2),
        "Config_D_Lab": res_d_lab, "Config_D_Runtime": res_d_rt_sig, "Gap_D": round(res_d_lab["bal_acc"] - res_d_rt_sig["bal_acc"], 2),
    },
    "high_class_table_4_4": {
        "Config_A": res_a_rt,
        "Config_B": res_b_rt,
        "Config_C_uncal_bs": res_c_rt_uncal["brier"],
        "Config_C": res_c_rt,
        "Config_D_uncal_bs": res_d_rt_uncal["brier"],
        "Config_D_sigmoid": res_d_rt_sig,
        "Config_D_isotonic": res_d_rt_iso
    },
    "gnn_table_4_5": {
        "GNN_Originaria": gnn_base_res,
        "GNN_Ottimizzata": gnn_opt_res
    },
    "feature_importance_4_3_2": {
        "imp_struct_raw_pct": imp_struct_raw_pct,
        "imp_struct_opt_pct": imp_struct_opt_pct,
        "imp_sem_opt_pct": imp_sem_opt_pct
    },
    "concept_drift_table_4_6": {
        "Static_Config_D": {
            "pre_drift": round(np.mean(static_f1s[:2]), 2),
            "post_drift": round(np.mean(static_f1s[2:4]), 2),
            "regime": round(np.mean(static_f1s[4:]), 2),
            "recall_high_end": static_rec_end
        },
        "Incremental_Non_Pesato": {
            "pre_drift": round(np.mean(unweighted_f1s[:2]), 2),
            "post_drift": round(np.mean(unweighted_f1s[2:4]), 2),
            "regime": round(np.mean(unweighted_f1s[4:]), 2),
            "recall_high_end": rec_u_end,
            "gate_interventions": gate_u
        },
        "HITL_Pesato_Entropia": {
            "pre_drift": round(np.mean(hitl_f1s[:2]), 2),
            "post_drift": round(np.mean(hitl_f1s[2:4]), 2),
            "regime": round(np.mean(hitl_f1s[4:]), 2),
            "recall_high_end": rec_h_end,
            "gate_interventions": gate_h
        }
    },
    "latency_demo_4_4_2": {
        "n_us_demo": 15,
        "lat_locale_ms": lat_locale_ms,
        "n_alert": n_alert_demo
    }
}

with open("ablation_results.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 80)
print("✅ ESPERIMENTI COMPLETATI! Risultati salvati in 'ablation_results.json'")
print("=" * 80)
print(json.dumps(report, indent=2))