#!/usr/bin/env bash
# fix_install.sh — Clears all stale minigpt/src/minigpt_core editable registrations
# and performs a clean reinstall.
#
# Usage (from project root, venv activated):
#   bash fix_install.sh

set -e
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}=== MiniGPT Clean Install Fix ===${NC}\n"

# ── 1. Confirm we're in the right directory ───────────────────────────────────
if [ ! -f "main.py" ] || [ ! -d "minigpt_core" ]; then
    echo -e "${RED}ERROR: Run this from the minigpt project root."
    echo -e "       Expected to find main.py and minigpt_core/ here.${NC}"
    exit 1
fi
echo -e "✓ Project root confirmed: $(pwd)"

# ── 2. Confirm venv is active ─────────────────────────────────────────────────
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${RED}ERROR: No venv active. Run: source venv/bin/activate${NC}"
    exit 1
fi
echo -e "✓ Venv active: $VIRTUAL_ENV"

# ── 3. Uninstall any existing minigpt registration ───────────────────────────
echo -e "\n${YELLOW}Step 1/5: Uninstalling any existing minigpt...${NC}"
pip uninstall minigpt -y 2>/dev/null && echo "  Uninstalled minigpt" || echo "  (not installed — skipping)"

# ── 4. Remove ALL stale .egg-link and direct_url files ───────────────────────
echo -e "\n${YELLOW}Step 2/5: Removing stale .pth, .egg-link, and dist-info files...${NC}"

SITE_PACKAGES="$VIRTUAL_ENV/lib/python$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages"
echo "  site-packages: $SITE_PACKAGES"

# Remove egg-links
find "$SITE_PACKAGES" -name "minigpt*.egg-link" -delete -print 2>/dev/null || true
find "$SITE_PACKAGES" -name "src*.egg-link"     -delete -print 2>/dev/null || true

# Remove dist-info directories
find "$SITE_PACKAGES" -maxdepth 1 -name "minigpt*.dist-info" -type d -exec rm -rf {} + 2>/dev/null || true
find "$SITE_PACKAGES" -maxdepth 1 -name "src*.dist-info"     -type d -exec rm -rf {} + 2>/dev/null || true

# Remove stale .pth files that point to old locations
for pth in "$SITE_PACKAGES"/__editable__*.pth "$SITE_PACKAGES"/easy-install.pth; do
    if [ -f "$pth" ]; then
        echo "  Found .pth: $pth"
        cat "$pth"
    fi
done

# Remove editable .pth files for minigpt/src
find "$SITE_PACKAGES" -name "__editable__minigpt*" -delete -print 2>/dev/null || true
find "$SITE_PACKAGES" -name "__editable__src*"     -delete -print 2>/dev/null || true

echo -e "  ${GREEN}Done cleaning stale files.${NC}"

# ── 5. Delete local __pycache__ and .egg-info ────────────────────────────────
echo -e "\n${YELLOW}Step 3/5: Cleaning local build artifacts...${NC}"
find . -name "__pycache__" -type d -not -path "./venv/*" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.egg-info"  -type d -not -path "./venv/*" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc"                -not -path "./venv/*" -delete              2>/dev/null || true
find . -name ".pytest_cache" -type d -not -path "./venv/*" -exec rm -rf {} + 2>/dev/null || true
echo -e "  ${GREEN}Done.${NC}"

# ── 6. Delete stale src/ if it still exists ───────────────────────────────────
if [ -d "src" ]; then
    echo -e "\n${YELLOW}Step 4/5: Removing stale src/ directory...${NC}"
    rm -rf src/
    echo -e "  ${GREEN}Removed src/${NC}"
else
    echo -e "\n${YELLOW}Step 4/5: No stale src/ found — skipping.${NC}"
fi

# ── 7. Fresh editable install ─────────────────────────────────────────────────
echo -e "\n${YELLOW}Step 5/5: Fresh editable install...${NC}"
pip install -e . --no-build-isolation

# ── 8. Verify ─────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}=== Verifying install ===${NC}"
python - << 'PYEOF'
import sys, os
# Ensure project root is first
sys.path.insert(0, os.getcwd())

try:
    import minigpt_core
    print(f"  minigpt_core location : {minigpt_core.__file__}")

    from minigpt_core.model import MiniGPT, DEFAULT_CONFIG
    m = MiniGPT(DEFAULT_CONFIG)
    params = m.count_parameters()
    size   = DEFAULT_CONFIG.estimated_size_mb(8)
    print(f"  MiniGPT import        : OK")
    print(f"  Parameters            : {params:,}")
    print(f"  INT8 size estimate    : {size:.2f} MB")
    print(f"  Under 16MB            : {'✓ YES' if size < 16 else '✗ NO'}")
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)
PYEOF

echo -e "\n${GREEN}=== Fix complete. Run: python main.py info ===${NC}"
