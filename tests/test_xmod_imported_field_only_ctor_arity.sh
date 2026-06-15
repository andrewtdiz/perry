#!/bin/bash
# Regression: imported classes with fields and no own constructor still expose
# the standalone constructor arity needed to forward args to an ancestor ctor.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PERRY="${PERRY_BIN:-${PERRY:-$REPO_ROOT/target/release/perry}}"
if [[ ! -x "$PERRY" ]]; then
  PERRY="$REPO_ROOT/target/debug/perry"
fi
if [[ ! -x "$PERRY" ]]; then
  echo "SKIP: perry binary not found (build with cargo build -p perry)"
  exit 0
fi
if [[ "$PERRY" != /* ]]; then
  PERRY="$(cd "$(dirname "$PERRY")" && pwd)/$(basename "$PERRY")"
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

cat > "$TMPDIR/root.ts" <<'TS'
export class Root {
  constructor(value: any) {
    (this as any).rootValue = value;
  }
}
TS

cat > "$TMPDIR/field_mid.ts" <<'TS'
import { Root } from "./root.ts";

export class FieldMid extends Root {
  midField: any = "mid";
}
TS

cat > "$TMPDIR/private_mid.ts" <<'TS'
class PrivateRoot {
  constructor(value: any) {
    (this as any).privateRootValue = value;
  }
}

export class PublicFieldMid extends PrivateRoot {
  publicMidField: any = "public-mid";
}
TS

cat > "$TMPDIR/main.ts" <<'TS'
import { FieldMid } from "./field_mid.ts";
import { PublicFieldMid } from "./private_mid.ts";

const direct: any = new FieldMid("direct");
console.log("direct root", direct.rootValue);
console.log("direct mid", direct.midField);

class ExplicitLeaf extends FieldMid {
  constructor(value: any) {
    super(value);
    (this as any).leafReady = "explicit";
  }
}

const explicit: any = new ExplicitLeaf("explicit");
console.log("explicit root", explicit.rootValue);
console.log("explicit mid", explicit.midField);
console.log("explicit leaf", explicit.leafReady);

class DefaultLeaf extends FieldMid {}

const defaultLeaf: any = new DefaultLeaf("default");
console.log("default root", defaultLeaf.rootValue);
console.log("default mid", defaultLeaf.midField);

const privateDirect: any = new PublicFieldMid("private-direct");
console.log("private direct root", privateDirect.privateRootValue);
console.log("private direct mid", privateDirect.publicMidField);

class PrivateDefaultLeaf extends PublicFieldMid {}

const privateDefault: any = new PrivateDefaultLeaf("private-default");
console.log("private default root", privateDefault.privateRootValue);
console.log("private default mid", privateDefault.publicMidField);
TS

cd "$TMPDIR"
"$PERRY" compile --no-cache --no-auto-optimize main.ts --output test_bin >/dev/null
RUN_OUTPUT="$(./test_bin 2>&1)"

EXPECTED="direct root direct
direct mid mid
explicit root explicit
explicit mid mid
explicit leaf explicit
default root default
default mid mid
private direct root private-direct
private direct mid public-mid
private default root private-default
private default mid public-mid"

if [[ "$RUN_OUTPUT" == "$EXPECTED" ]]; then
  echo "PASS"
  exit 0
fi

echo "FAIL: imported field-only constructor did not preserve forwarded args"
echo "Expected:"
echo "$EXPECTED"
echo ""
echo "Got:"
echo "$RUN_OUTPUT"
exit 1
