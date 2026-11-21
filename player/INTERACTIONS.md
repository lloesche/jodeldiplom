# How the player's controls interact

What each control does to the others, above all to a transition. The
tests named after each rule hold it; a change that breaks a rule fails
them.

A transition goes through these states:

| State | Meaning |
|---|---|
| set up | shown on the Transition tab, not sent |
| armed | taken by the engine, waiting for its window |
| preparing | about 4 s before the window: both tracks onto their stems, the next one decoding ahead |
| mixing | the window is heard |
| settling | the next track plays on alone, its tempo straightening |
| done | |

On the Transition tab, the first three show as `enter sets the cues` / `g arms`
/ `armed`, and the last two as `mixing`.

## Stems

- **S1.** Stems can be switched at any time, in any state, and never call
  a transition off. They never change when it starts or ends, its window,
  where the next track comes in, its pairing or its tempo. They change
  only which stems are heard: while armed, the playing track's; while
  preparing, mixing or settling, both tracks', in their fades.
  - `stems_switched_during_a_transition_go_into_its_fades`
  - `stems_switched_while_armed_leave_the_transition_where_it_was`
  - `stems_switched_at_a_bar_in_the_window_change_there`
  - `stems_switched_while_settling_go_into_the_next_tracks_fades`
  - `stem_toggles_never_touch_the_transition` (player)
- **S2.** A stem change is heard when it would be without a transition: on
  the next bar the engine can make (or at once, per the setting), on the
  playing track's bars.
  - `stems_switched_at_a_bar_in_the_window_change_there`
- **S3.** Switching stems never moves playback: no jump, no restart, no
  gap. A stem goes in and out over 10 ms, and a route change crossfades
  over 20 ms.
  - the stems and switch tests
- **S5.** A track got ready plays its kept stems: nothing is separated
  while it plays (on stems, from a jump, in a transition or the
  Transition tab). Only a track whose stems are not kept is separated live.
  - `kept_stems_play_without_a_separation` (engine)
- **S4.** After a transition, the next track's stems follow the settings,
  whatever they were during it.
  - `after_a_transition_the_next_tracks_stems_follow_the_settings`

## Transition keys

- **T0.** `enter` (and a double click) on the Transition tab never moves
  playback: it sets the transition's cue on the track focused, over the
  bars selected, else eight bars from the cursor's bar. The fades show on
  a track once it has a cue; `p` without one only names the preset in the
  header. The window starts at the playing track's cue and the next track
  comes in at its own; it lasts as long as the longer cue (of those
  selected, if either was), counted in the playing track's bars.
  - `g_arms_a_transition_from_the_cue_into_the_cue`
  - `the_cues_decide_the_window_and_an_armed_one_follows_them`
  - `the_transition_cursor_moves_by_bars_and_cues_one`
  - `the_fade_curves_are_drawn_in_the_terminal_lanes` (view)
- **T1.** `p`, `m`, `r` and the fades edited (`e`) change a transition
  set up or armed. Armed, it is sent again with the same window and
  entry, and keeps them, also while preparing; a new cue moves it to the
  cue. During the mix they do nothing, and say so.
  - `transition_keys_wait_while_mixing` (player)
  - `a_plan_changed_while_preparing_keeps_its_window`
- **T2.** `g` arms at the two cues, only once both tracks have one and
  the playing track's has not been heard yet, and disarms. `esc` disarms
  (after clearing a selection); with nothing armed it clears the focused
  track's cue. During the mix, `g`, `enter` and `esc` do nothing, and
  say so: a mix under way is never cut off by a transition key.
  - `transition_keys_wait_while_mixing` (player)
  - `calling_off_a_mix_under_way_does_nothing`
- **T3.** Only moving playback calls a transition off: a seek, the next
  or previous track, playing another track, stopping, a new queue, or a
  queue change that takes away its next track. Pausing does not; the
  transition goes on after the pause. The next track key during the mix
  cuts into the next track where it is heard: the playing track stops
  there and the next one plays on alone (it never starts over).
  - `a_paused_transition_goes_on_after_the_pause`
  - `next_while_a_transition_mixes_cuts_into_the_next_track_where_it_is`
  - `n_mixes_into_the_next_track_by_the_transition_tabs_preset` (player)
  - the transition and queue tests
- **T4.** Settings other than the stems never call a transition off.
  Armed, the transition takes them when it prepares. During it, they
  apply once it is over (a notice says so).
  - `other_settings_during_a_mix_wait_for_its_end`
- **T5.** Whatever ends a transition (done, called off, the output device
  switched, the engine restarted) leaves the Transition tab with no transition
  armed or mixing. A word from the engine about one that is gone is not
  taken.
  - `g_arms_a_transition_from_the_cue_into_the_cue`
  - `the_transition_cursor_moves_by_bars_and_cues_one`
  - `an_engine_restart_leaves_no_transition_behind` (player)
- **T6.** `esc` never leaves the Transition tab: only `tab` and the tab buttons
  do.
  - `the_transition_cursor_moves_by_bars_and_cues_one`

## Jumps

- **J1.** Bar jumps (`1`–`9` when the bar heard ends, with shift at once) are refused while
  a transition is armed or under way. A jump would move the playing
  track's time, which the window is set in.
  - `bar_jumps_wait_for_no_transition` (player)
- **J2.** A jump never leaves a gap, and lands on its bar, also with stems
  off.
  - the jump tests

## Queue

- **Q1.** Removing or moving tracks never interrupts what plays. The one
  playing, removed, gives way to the next. A transition goes on while its
  next track is still the one after the playing one, and is called off
  when it is not.
  - the queue tests
  - `a_transition_follows_its_tracks_through_the_queue`
- **Q2.** Saving the session or a queue never changes playback.
- **Q3.** A track in the queue shows a spinner while it is worked on, a
  dot while work for it is still to come, and nothing once all of it is
  in the caches (its analysis and waveforms, its lyrics looked up and
  timed). What is in the cache is handed over at once when it is wanted
  (the Transition tab's lanes, Karaoke's lines), never behind an analysis under
  way.
  - `the_cache_answers_while_an_analysis_is_under_way`
  - `queue_side` (the snapshot)

## Karaoke

- **K1.** Karaoke mode exists only fullscreen (the whole terminal in a
  terminal). On the Karaoke tab, as in the mode, every stem plays and the
  voice at its slider's level, moved at once, so what is heard, metered and
  recorded there is what the mode plays; leaving the tab, by any way, puts
  the stems back as the settings have them, and coming back plays the voice
  at its level again. Without a GPU the stems stay as they are. `esc` on the
  tab does nothing.
  - `the_tab_plays_the_voice_at_its_level_and_esc_does_nothing_there`
  - `without_a_gpu_the_tab_leaves_the_stems_alone`
- **K1b.** A song is sung only once its lyrics are ready: found and their
  words timed (or found where they cannot be timed: no speech model, or
  the timing failed). Until then, and for a song with none (none found, an
  instrumental, the look-up failed), the tab's button is greyed out with
  why beside it, entering the mode is refused, and in the mode START is
  greyed out and refused (`N` goes on to the next song); the status line
  says why.
  - `no_karaoke_for_a_song_until_its_lyrics_are_ready`
- **K2.** `esc` is the only way out of the mode (`tab`, the screens and the
  palette's tabs are not reached from it). Going in stops the song playing
  (or paused) and holds it at its start, START showing at once; going out
  never moves playback and comes back to the tab, the voice still at its
  level. Going in with shift (`shift+enter`, shift+click on the button)
  leaves the song where it is: nothing stops or rewinds, no count-in (for
  trying things out).
  - `the_mode_plays_the_voice_at_its_level_and_esc_keeps_it_on_the_tab`
  - `shift_enter_goes_into_the_mode_where_the_song_is`
- **K3.** In the mode, a song's end holds the next song at its start: only
  START (`enter`, `space`, the button) plays it, after its count-in. `N`
  holds the next song for START. The mode arms no transitions (one armed
  is called off going in), and `g`, bar jumps and the seek keys do nothing
  there; the controls' seek bar seeks, except while a song is held for
  START or counting in. Nothing plays in the mode without a count-in:
  `space` pauses a song playing and is START otherwise, and a paused song
  goes on at START only after the same count-in.
  - `holding_stops_after_a_track_and_names_the_next` (engine)
  - `the_mode_arms_no_transitions_and_makes_no_jumps`
  - `next_in_the_mode_holds_the_next_song_for_start`
- **K3b.** The mode's lines never move for its controls: those come up
  over the picture's bottom (the voice slider, the seek bar, the keys,
  START), while the pointer moves in the bottom quarter of the screen (for
  3 s after it last moved there; at once gone above it), while a slider or
  the seek bar is held, after a key moved the slider, and while a song
  waits for START.
  - `the_controls_come_up_near_the_bottom_over_the_lines_with_a_seek_bar`
  - `space_never_plays_a_song_without_its_count_in`
- **K4.** A change of the voice's level never moves playback and never
  touches a transition (as S1). In the mode it is heard at once.
  - `the_mode_turns_the_voice_down_and_esc_brings_it_back`
  - `a_stem_at_a_level_is_heard_but_not_full` (engine)
- **K5.** The count-in is "3", "2", "1", a second each, and "1" ends as the
  song's first word is sung (its words timed; else at its start). Before
  it, from START, the song's title and artist show alone for 2 s and fade
  out over 1 s; then the first line alone until it is sung. A song sung
  from 6 s in or later plays at once (title and count over its intro);
  one sung sooner starts that much later, after silence.
  It follows the audio clock, drawn from the positions heard. The lines
  stay readable under the numbers. `esc` in the silence before the song
  calls the start off, and the song is held at its start again; in its
  intro the song plays on, and `space` pauses it there.
  - `start_counts_in_three_seconds_whatever_the_song`
  - `the_count_in_ends_as_the_first_word_is_sung`
  - `the_title_fades_before_the_count_and_the_music_waits_for_it`
  - `lyrics_are_asked_for_on_the_tab_and_follow_the_song`
  - `a_count_in_is_silence_then_the_track_heard_from_its_end` (engine)
  - `esc_during_a_count_in_calls_the_start_off`
- **K5b.** The count-in comes again, whatever started the song (the mode
  entered with shift, `N`, a seek): before its first sung line, as large as
  START's, and before the singing comes back after a break of 8 s or more,
  half as large in the same place at the top: apart from the lines, never
  moving with them. Whole or not at all: where
  the song has been heard from without a jump (the mode entered there, a
  seek, another song, a pause ended) less than three seconds before the
  words, there is none. Never two counts at once.
  - `a_long_break_counts_in_again_whole_or_not_at_all`
- **K6.** The microphone is open while the Karaoke tab or the mode shows,
  and closed elsewhere. It is listened to, to hear the singer, and never
  played: it has no level, and nothing of it reaches the speakers. To take
  the music's echo out of it, it sees what the device plays, buffer by
  buffer (a tap at the output's last step, set on every output opened:
  another rate, another device, an engine restarted). `i` takes the next
  input, kept in the settings.
  - `the_microphone_goes_into_each_output` (engine)
  - `the_microphone_is_open_while_karaoke_shows`
- **K16.** The microphone never changes what plays: with it open, tapped,
  singing, its echo canceller on or off, the device gets exactly the bytes
  the engine wrote, in every format.
  - `a_tap_never_changes_what_plays` (output)
  - `the_cleaner_takes_the_echo_of_what_plays_out_of_the_voice` (output)
- **K11.** The words light up `karaoke_lead` (200 ms by default) before
  the voice sings them, so they can be read first: the lines, their fill
  and their scrolling all run that much ahead of what is heard.
  - `lyrics_are_asked_for_on_the_tab_and_follow_the_song`
- **K7.** Lyrics are looked up while Karaoke shows: the song and the next
  two. Only their words are taken; any times they come with are dropped
  (never shown, never kept, never an anchor). Once our word timing has
  them, a line comes up two seconds before its first word, is sung from
  its first word to its last (filling word by word), and a pause of more
  than four seconds shows `♪`; two seconds before the pause ends the next
  line moves up into its place. Lines only ever move up.
  - `after_a_pause_the_lines_go_on_up`
  - `lyrics_are_asked_for_on_the_tab_and_follow_the_song`
- **K8.** The speech model and the speech recogniser are got before the
  player starts, by its setup screen (`crate::setup`), into `models` in the
  data folder, each checked against the length and SHA-256 built in (the
  recogniser's checkpoint downloaded and checked, packed at 4 bits, the
  packed file checked, the checkpoint removed); with both there and
  checked, the player starts at once and never goes online for them. The
  screen cannot be skipped: a failure offers only Try again and Quit.
  - `everything_is_got_checked_and_stamped_then_nothing_more`
  - `offline_it_waits_for_try_again_and_goes_on_from_there`
  A text from LRCLIB that does not fit what
  the recogniser hears is swapped for one of LRCLIB's others that does
  (`a_fetched_text_that_does_not_fit_is_swapped_for_one_that_does`). The words of lyrics
  found are timed in the background (the voice separated on the GPU, the
  model on the GPU or half the processor's threads, the whole text aligned
  at once) and kept in the cache; the lines show as soon as the text is
  found and fill word by word once timed. A song no longer wanted is not
  delivered. A player built without it shows the words untimed and says
  so on the tab. Lyrics kept by an older player (with times that came with
  them) are deleted when read and looked up again.
  - `a_line_lasts_until_its_words_are_sung`
  - `the_speech_model_is_found_where_the_build_put_it`
- **K9.** In the mode, lines that change move: for 0.3 s each goes from
  where it was to where it goes (its size and colour too), a new one rises
  in fading in, one that leaves rises away fading out. A frame redraws only
  the text that changed.
  - `the_lines_scroll_from_where_they_were`
  - `a_frame_drawn_where_it_changed_is_the_frame_drawn_afresh`
  - `a_line_filling_redraws_only_itself`
- **K12.** The music's echo is taken out of what the microphone hears
  (`karaoke_echo`, on by default), and the singer never is: the voice it
  gives is the microphone less the echo the music predicts, nothing more.
  The meter shows the singer isolated from that voice (`jodel_output::isolate`:
  what the canceller leaves of the band and the room's own sounds, a fire, a
  fan, taken out by a small network trained on singing, 11 to 21 ms late;
  the voice kept stays the canceller's); the tab says how much echo goes and how late
  it comes once learnt (within a second or so of music, however loud).
  With the music alone the meter stays down by more than 20 dB through a
  laptop's speakers; every syllable of a voice sung over it, from twice as
  loud as the echo to 24 dB under it, comes through whole. A held note
  teaches the canceller nothing; the volume turned up is learnt again
  within seconds. `r` records 30 s of what the canceller works with, for
  tuning it. The voice is kept block by block with where in the song it
  was sung, for scoring the singing.
  - `the_cleaner_takes_the_echo_of_what_plays_out_of_the_voice` (output)
  - `the_echo_goes_and_a_voice_over_it_stays` (output, `aec`)
  - `a_laptop_in_a_living_room`, `a_wireless_speaker` (output, `aec`)
  - `a_singer_through_a_laptop`, `a_held_note_never_spoils_the_filter`,
    `the_volume_turned_up_is_learnt_again` (output, `aec`)
  - `a_voice_heard_back_is_taken_out`, `the_howl_guard_holds_feedback_down`,
    `a_held_note_is_not_a_howl` (output, `aec`)
  - `the_tab_says_when_feedback_is_held_down` (view)
  - `r_records_the_echo_canceller`
- **K13.** The Transition tab asks for its tracks' lyrics; once timed,
  their words go across the middle of the `vocals` lane, each where it
  is sung, each letter where the aligner heard it (else spread over the
  syllables), a line where a sound is held, the one heard in the accent
  colour. Where a word stands never depends on which others are in view,
  so scrolling never moves them. `L` turns them off (no lyrics asked
  for) and on, kept in the settings.
  - `the_words_go_on_the_voice_lane_where_they_are_sung`
  - `crowded_words_keep_their_places_as_the_view_scrolls` (`view::lanes`)
  - `the_lane_labels_stand_where_their_words_are_sung` (`view::lanes`)
  - `heard_letters_stand_where_they_are_sung` (`view::lanes`)
  - `the_path_finds_planted_word_times_exactly` (lyrics, the letters)
- **K15.** `z` (or `z fit lyrics` in the Transition tab's bottom border)
  zooms the track selected, and only it, to the widest zoom that leaves
  out no word of its lyrics; without timed lyrics it says so and keeps
  the zoom. The Karaoke tab shows the playing song's voice with its
  words where the other tabs have the spectrum, each where it is sung,
  at the zoom that leaves none out (fixed for the song).
  - `z_zooms_the_selected_track_to_fit_its_lyrics`
  - `the_karaoke_tab_shows_the_voice_with_every_word`
- **K14.** While the Karaoke tab shows with the microphone open, the window
  keeps drawing (the meter moves with the voice), whether a song plays or
  not.
  - `the_microphone_is_open_while_karaoke_shows`
- **K10.** A microphone giving only exact zeros for 1.5 s is shown as
  silent, with what to check (on macOS: the system's permission).
  - `a_microphone_giving_only_zeros_is_told_apart` (output)
  - `lyrics_found_come_again_with_their_words_timed_and_kept`
