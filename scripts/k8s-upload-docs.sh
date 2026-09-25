#!/usr/bin/env bash
# Upload the course material from this PC into the iu-agent-docs volume of the Kubernetes deployment.
#
#   scripts/k8s-upload-docs.sh [SOURCE_DIR] [NAMESPACE]
#
# Only file types the agent can index are sent (PDF, DOCX, PPTX, XLSX, HTML, notebooks, text),
# the folders from IU_EXCLUDE_GLOBS (Bill, Certificate) are skipped. Works from Git Bash on Windows,
# macOS and Linux; needs kubectl and GNU tar. When done it writes /data/iu/.upload-complete, which
# releases the waiting ingest Job.
set -euo pipefail

SRC="${1:-${IU_DOCS_PATH:-$HOME/OneDrive/IU}}"
NS="${2:-iu-agent}"
TARGET="deploy/iu-agent"

if [ ! -d "$SRC" ]; then
  echo "source folder not found: $SRC" >&2
  exit 1
fi

echo "waiting for the agent pod in namespace $NS ..."
kubectl -n "$NS" rollout status "$TARGET" --timeout=300s

echo "collecting files under $SRC ..."
cd "$SRC"
find . -type f \
  \( -iname '*.pdf' -o -iname '*.docx' -o -iname '*.pptx' -o -iname '*.xlsx' -o -iname '*.md' \
     -o -iname '*.markdown' -o -iname '*.rst' -o -iname '*.txt' -o -iname '*.html' -o -iname '*.htm' \
     -o -iname '*.ipynb' -o -iname '*.tex' -o -iname '*.bib' -o -iname '*.py' -o -iname '*.csv' \) \
  -not -path './Bill/*' -not -path './Certificate/*' -not -path '*/.git/*' -not -path '*/.venv/*' \
  -not -name '.*' -not -name '~$*' > "${TMPDIR:-/tmp}/iu-upload-list.txt"
COUNT=$(wc -l < "${TMPDIR:-/tmp}/iu-upload-list.txt")
echo "uploading $COUNT files to $TARGET:/data/iu ..."

kubectl -n "$NS" exec -i "$TARGET" -- sh -c 'rm -f /data/iu/.upload-complete && mkdir -p /data/iu && tar -C /data/iu -xf -' \
  < <(tar -cf - --files-from="${TMPDIR:-/tmp}/iu-upload-list.txt")

kubectl -n "$NS" exec "$TARGET" -- sh -c 'touch /data/iu/.upload-complete && du -sh /data/iu'
echo "done. The ingest Job starts indexing now:  kubectl -n $NS logs -f job/iu-agent-ingest"
