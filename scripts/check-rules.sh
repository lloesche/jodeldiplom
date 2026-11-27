#!/usr/bin/env bash
# The project's rules that the compiler does not hold, checked over the
# tracked files:
#
# - a word that stays out of this project is nowhere, in no file's text
#   (as it is in the working tree) and no file's name; binary files are
#   passed over (`git grep -I`), where its three letters turn up by chance
#   in random bytes;
# - every text file is stored with LF line endings (in the index: as it
#   is committed);
# - no shader writes one lane of a vector by index (Apple's GPUs may write
#   the whole vector back: gpu/README.md, "Rules for kernels");
# - big source files are reported, not failed: over 1000 lines of code is
#   a question for review (split by concern where that helps), not a limit.
#
# That every file is written through the durable crate is clippy's to
# hold (clippy.toml's disallowed-methods).
#
# Run it from anywhere in the repository: `bash scripts/check-rules.sh`.
# It needs bash, git and the POSIX tools (awk, grep, sort, xargs), GNU or
# BSD: Linux, macOS, Git Bash on Windows. Exit status 1 if a rule fails.
set -u

root=$(git rev-parse --show-toplevel) || exit 2
cd "$root" || exit 2
failed=0

fail() {
    printf 'FAIL: %s\n' "$*"
    failed=1
}

# git as the checks call it: paths printed as they are.
g() {
    git -c core.quotePath=false "$@"
}

# The word, spelled so that this file does not hold it.
word='m''q''a'

echo "== The word"
# git grep: 0 found, 1 nothing found, more an error.
hits=$(g grep -I -n -i -e "$word")
case $? in
0)
    printf '%s\n' "$hits"
    fail "the word is in the files above"
    ;;
1) ;;
*) fail "git grep failed" ;;
esac
if files=$(g ls-files); then
    if names=$(printf '%s\n' "$files" | grep -i -e "$word"); then
        printf '%s\n' "$names"
        fail "the word is in the file names above"
    fi
else
    fail "git ls-files failed"
fi

echo "== Line endings"
# `git ls-files --eol`: the index's endings first (`i/crlf`, `i/mixed`
# have a CR LF), the working tree's, the attributes, a tab, the path.
if eol=$(g ls-files --eol); then
    if crlf=$(printf '%s\n' "$eol" | grep -E '^i/(crlf|mixed)'); then
        printf '%s\n' "$crlf"
        fail "the files above are stored with CR LF line endings (want LF)"
    fi
else
    fail "git ls-files --eol failed"
fi

echo "== Shader lane writes"
# A store into one lane of a vector by index (`a[i][j] = v`, `a[i].x = v`)
# in any WGSL: on Apple's GPUs it may write the whole vector back, over
# what other invocations wrote to its other lanes (gpu/README.md, "Rules
# for kernels"). Shared arrays and buffers are f32s, written whole.
hits=$(g grep -n -E -e '\][[:space:]]*\[[^]]+\][[:space:]]*[-+*/]?=[^=]' \
    -e '\]\.[xyzw][[:space:]]*[-+*/]?=[^=]' -- '*.wgsl')
case $? in
0)
    printf '%s\n' "$hits"
    fail "the shader lines above write one lane of a vector (write f32s whole)"
    ;;
1) ;;
*) fail "git grep failed" ;;
esac

echo "== Big files (a question for review, not a failure)"
# Lines of code of each tracked source file: tests (a tests folder, a
# tests.rs) not counted, nor a file's `#[cfg(test)] mod tests` (its
# attributes too) and what follows it. Those over 1000, largest first; a
# notice each on GitHub.
g ls-files -z -- '*.rs' '*.wgsl' '*.sh' '*.py' |
    xargs -0 awk '
        function done_file() { if (f != "" && n > 1000) printf "%6d  %s\n", n, f }
        FNR == 1 {
            done_file()
            f = FILENAME; n = 0; at = 0
            skip = (f ~ /(^|\/)tests\// || f ~ /(^|\/)tests\.rs$/)
            if (skip) f = ""
        }
        skip { next }
        # `#[cfg(test)]`, maybe more attributes, then `mod tests`: the
        # count ends before the first of them.
        /^#\[cfg\(test\)\]/ { if (!at) at = n + 1; n++; next }
        at && (/^#\[/ || /^[[:space:]]/ || /^\)\]/) { n++; next }
        at && /^(pub(\([a-z]+\))? )?mod tests/ { n = at - 1; skip = 1; next }
        { at = 0; n++ }
        END { done_file() }' |
    sort -rn |
    while read -r lines path; do
        printf '%6d  %s\n' "$lines" "$path"
        if [ -n "${GITHUB_ACTIONS:-}" ]; then
            printf '::notice file=%s::%s lines of code: a question for review\n' "$path" "$lines"
        fi
    done

if [ "$failed" -ne 0 ]; then
    echo "Some rules failed."
    exit 1
fi
echo "All rules hold."
