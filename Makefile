# The player for the platform make runs on, from this folder:
#
#   make jodeldiplom   (or just make)   the player, jodeldiplom; on macOS
#                                      as the app "Jodeldiplom.app" (signed
#                                      ad hoc). The speech models are not
#                                      part of the build: the player gets
#                                      and checks them itself on its first
#                                      start (its setup screen), into its
#                                      data folder
#   make diag                          builds the player and runs its
#                                      --diag: what the player finds on
#                                      this system, for a bug report
#   make clean-cache                   forgets the beat analyses, the kept
#                                      stems and the lyrics (the words
#                                      found and our timings of them): each
#                                      track is analysed, separated, its
#                                      lyrics looked up and timed again
#                                      when it is wanted (the player's
#                                      --clean-cache: it knows where its
#                                      data folder is)
#   make clean-diag                    forgets the record of the last
#                                      analyses that --diag shows
#   make clean                         removes the build output, and does
#                                      clean-cache and clean-diag; the
#                                      models and the settings stay
#
# Everything lands in target/release (or $CARGO_TARGET_DIR/release). The
# player's fonts come from player/fonts. CARGO picks another cargo, e.g.
# make CARGO="cargo +1.100.0".
#
# The jobs print to the terminal make runs in on every system: the
# release player on Windows has no console of its own and attaches to
# that one.

CARGO ?= cargo
TARGET_DIR := $(or $(CARGO_TARGET_DIR),target)
RELEASE := $(TARGET_DIR)/release

ifeq ($(OS),Windows_NT)
PLATFORM := Windows
EXE := .exe
else
PLATFORM := $(shell uname -s)
EXE :=
endif

PLAYER := $(CARGO) run --release --locked -q -p jodeldiplom --

.PHONY: all jodeldiplom diag clean-cache clean-diag clean

all: jodeldiplom

jodeldiplom:
	$(CARGO) build --release --locked -p jodeldiplom
ifeq ($(PLATFORM),Darwin)
	bash player/macos/make-app.sh "$(RELEASE)/jodeldiplom" player/fonts \
		"$$($(CARGO) pkgid -p jodeldiplom | sed 's/.*[#@]//')" "$(RELEASE)"
	@echo "Built $(RELEASE)/Jodeldiplom.app (and $(RELEASE)/jodeldiplom)"
else
	@echo "Built $(RELEASE)/jodeldiplom$(EXE)"
endif

diag:
	$(PLAYER) --diag

clean-cache:
	$(PLAYER) --clean-cache

clean-diag:
	$(PLAYER) --clean-diag

clean: clean-cache clean-diag
	$(CARGO) clean
