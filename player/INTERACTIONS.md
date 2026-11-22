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

The stems are strips of the desk inside the player's (the Mix tab); the
stem keys are their mutes.

- **S1.** Stems can be switched at any time, in any state, and never call
  a transition off. They never change when it starts or ends, its window,
  where the next track comes in, its pairing or its tempo. They change
  only which stems are heard: outside a transition and while it is armed,
  on the desk; while preparing, mixing or settling, in both tracks' fades
  too (the engine mixes them with the strips' gains).
  - `stems_switched_during_a_transition_go_into_its_fades` (engine)
  - `stems_switched_while_armed_leave_the_transition_where_it_was` (engine)
  - `stems_switched_at_a_bar_in_the_window_change_there` (engine)
  - `stems_switched_while_settling_go_into_the_next_tracks_fades` (engine)
  - `stem_toggles_never_touch_the_transition` (player)
- **S2.** A stem key's change lands on the first beat of the next bar at
  least 0.1 s ahead, to the frame of the player's output (or at once, per
  the setting, or without bars or a position); pressed again before that
  bar, the change is taken back there. `alt+0` puts every stem's strip
  back as it starts, its mutes on the bar too. The bar is a second ahead
  at least while a transition is armed or heard, and 0.75 s while the
  playing track's stems are not there yet and must be separated on the GPU
  first (none kept: past the separator's first chunk); kept stems, or ones
  there already, need no more than 0.1 s. Its frame comes from the last
  two positions told (none after a seek, a jump or another track until two
  more), so it is the frame heard there.
  - `stems_switch_on_the_next_bar_as_mutes_of_their_strips`
  - `a_first_touch_lands_on_a_bar_its_stems_will_meet`
  - `a_stem_muted_on_its_frame_is_heard_and_unmuted_the_file_is_again`
  - `stems_switch_from_keys_and_buttons_with_a_gpu`
- **S3.** Switching stems never moves playback: no jump, no restart, no
  gap. A strip's change is a 20 ms ramp, and the desk crosses between the
  file and the stems over 10 ms (linearly: they are the same music).
  - `a_stem_muted_on_its_frame_is_heard_and_unmuted_the_file_is_again`
  - the mixer's lanes tests (output)
- **S4.** With every stem's strip as it starts the desk plays the file,
  bit for bit; the stems are heard in its place only while one is
  touched, and the file again once none is and the ramps have settled.
  - `a_stem_muted_on_its_frame_is_heard_and_unmuted_the_file_is_again`
  - `main_through_a_lane_input_is_bit_exact_with_stems_made` (engine)
- **S5.** The stems are made while a stem's strip is touched or the Mix,
  Karaoke or Transition tab (or the mode) shows: from the stems kept,
  else separated on the GPU; without one, only the kept ones. Otherwise
  none are made. A track got ready plays its kept stems: nothing is
  separated while it plays.
  - `the_stems_are_made_while_touched_or_a_tab_shows_them`
  - `kept_stems_play_without_a_separation` (engine)
- **S6.** A transition with stems touched never jumps: the playing track
  keeps its stems on the desk until it is on its stems in the engine (and
  its crossfade there is over), and the next track goes back onto its
  file only once its stems have been on the desk for the mixer's switch
  (2 s of waiting at most). A stem changed on a bar inside the window
  changes there; a stem soloed is heard alone throughout, the file never
  silent while it stands for the stems.
  - `a_transition_with_a_stem_muted_never_jumps` (engine)
  - `a_transition_with_a_stem_soloed_never_goes_silent` (engine)
  - `a_stem_muted_on_a_bar_in_the_window_goes_off_there` (engine)
  - `the_next_tracks_return_waits_until_its_stems_are_switched_to` (engine)
  - `a_part_soloed_keeps_main_while_main_stands_for_the_parts` (output)
- **S7.** After a transition the next track goes back onto its own route,
  and its stems are heard as the strips have them (on the desk, from its
  parts), whatever they were during it: what a transition mixes with
  them (`StemMask`) is the transition's alone.
  - `after_a_transition_the_next_track_plays_its_own_route_again` (engine)
  - `the_next_tracks_return_waits_until_its_stems_are_switched_to` (engine)

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
  terminal). On the Karaoke tab, as in the mode, the voice plays at its
  slider's level (the vocals' lane on the desk, over the VOCALS strip,
  which stays as set), moved at once, so what is heard, metered and
  recorded there is what the mode plays; leaving the tab, by any way, puts
  the lane back at full, and coming back plays the voice at its level
  again. Without a GPU only songs whose stems are kept have it turned
  down. `esc` on the tab does nothing.
  - `the_tab_plays_the_voice_at_its_level_and_esc_does_nothing_there`
  - `without_a_gpu_the_voice_turns_down_the_kept_stems`
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
  - `the_mode_plays_the_voice_at_its_level_and_esc_keeps_it_on_the_tab`
  - `a_stem_level_on_the_desk_is_heard_at_once_and_untouched_is_the_file_again` (engine)
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
  or the Mix tab with its MIC strip selected, and closed elsewhere (the
  Mix tab alone does not open it). It is listened to, to hear the singer, and never
  played: it has no level, and nothing of it reaches the speakers. To take
  the music's echo out of it, it sees what the device plays, buffer by
  buffer (a tap at the output's last step, set on every output opened:
  another rate, another device, an engine restarted). `i` takes the next
  input, kept in the settings.
  - `the_microphone_goes_into_each_output` (engine)
  - `the_microphone_is_open_while_karaoke_shows`
  - `the_faders_are_set_on_the_mixer_and_again_on_a_new_one`
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
- **K17.** How late the singer's voice comes is one value, for the scoring
  and what Keep records: the delay measured by a sync for the output and
  microphone in use, else the echo canceller's once steady (within 10 ms
  of itself for 3 s; a delay that jumps is never taken), else the one
  estimated from the singing, else 150 ms; the nudge (−200 to +400 ms in
  steps of 10, kept per output and microphone) on any of them. The tab's
  Delay row shows the value in force, its sync row where it comes from.
  Mid-song it moves at most 20 ms a second; a nudge or a sync is taken at
  once. Another output or microphone starts the estimate and the echo's
  wait over. Each level has its own share of a word to hit (easy a tenth,
  medium a quarter, hard two fifths).
  - `the_delay_in_force_is_the_first_there_is_and_says_where_it_comes_from`
  - `mid_song_the_delay_never_jumps_but_a_nudge_or_a_sync_does_at_once`
  - `another_pair_starts_the_estimate_over`
  - `singing_heard_120_ms_late_is_found_within_10_ms` (`core::lag::estimate`)
  - `each_level_asks_its_own_share_of_a_word` (`core::singing`)
- **K18.** `s` on the Karaoke tab (or the palette) syncs, with a
  microphone open (else it says one is needed) and the player's strip
  heard on the Mix tab (not muted or far down, nor another strip soloed:
  a stem's strip soloed silences the clicks too): the song stops, 8 clicks at 90 BPM play after a
  second of silence, on an output of their own (nothing else changes what
  plays), and a "ta" with each is timed in the frames the voice is scored
  in (the song's voice left before the microphone's tap moves to the
  clicks left out). Five answers within 150 ms of their median make the
  delay (their median), kept with `enter` for the output and microphone
  the sync started with (`delays.txt` beside the settings), the nudge back
  to 0; else "again?". `r` goes again, `esc` cancels (each a button too),
  leaving the tab calls it off, the output changing ends it.
  - `s_stops_the_song_and_plays_the_clicks_with_a_microphone_open`
  - `no_sync_without_a_microphone_or_where_the_clicks_are_not_heard`
  - `a_delay_measured_is_kept_for_its_output_and_microphone_the_nudge_back_to_0`
  - `a_delay_is_kept_for_the_pair_it_was_measured_with`
  - `too_few_answers_ask_for_another_go_and_leaving_the_tab_calls_it_off`
  - `the_panels_buttons_take_clicks_and_nothing_else_does`
  - `the_app_plays_the_clicks_and_hears_a_ta_after_each`
  - `the_output_changing_mid_sync_ends_it`
  - `the_songs_blocks_left_at_the_start_are_left_out` (`sync`)

## Mix

- **M1.** Bit-perfect is the default: every start has each strip at 0 dB,
  unmuted, nothing soloed, and nothing of the desk is kept between runs.
  The desk adds no gain until a fader is moved; the master never adds any
  (its VOLUME is the device's own control, after the samples).
  - `every_control_sends_its_change`
  - `a_ramp_lands_on_its_frame_and_unity_solo_is_bit_exact` (output)
- **M2.** A new mixer (another device, an engine restarted) gets the
  faders, mutes and solos as they were set this run.
  - `the_faders_are_set_on_the_mixer_and_again_on_a_new_one`
- **M3.** A fader, mute or solo never moves playback and never clicks: it
  is a 20 ms ramp in the mixer; the preview's crossfades go on under it
  (they are its players' own faders, times the strip's).
  - `mute_and_solo_silence_strips_without_a_click` (output)
- **M4.** A fader held follows the pointer anywhere until it is let go
  of, to less than a cell in the window; within 0.4 rows of 0 dB it is at
  exactly 0 dB (the detent), and a key step across 0 dB stops there; a
  double click is 0 dB (a fader moved since its press is no first click
  of one); the wheel is 1 dB a step.
  - `a_fader_follows_the_pointer_by_eighths_of_a_cell`
  - `a_press_on_the_track_brings_the_cap_and_a_double_click_is_unity`
  - `the_wheel_moves_a_fader_by_a_db_and_the_buttons_toggle`
  - `a_fader_flicked_and_taken_again_soon_is_no_double_click`
  - `keys_step_through_the_scale_to_off_and_back_and_stop_at_unity`
- **M6.** The player's stems have strips of their own beside it while its
  input has lanes; the keys and the mouse act on them as on any strip (the
  arrows through the strips drawn), and where the window is narrow the
  strips narrow, then the microphone's strip goes first, then the
  preview's, then the stems'; at 80 columns all but the microphone's.
  - `the_arrows_go_through_the_stems_only_while_the_player_has_them`
  - `a_stems_strip_takes_the_pointer_and_its_buttons`
  - `the_stems_sit_beside_the_player_under_their_bracket`
  - `the_mix_tab_is_a_desk` (view)
- **M5.** The meters are read once a drawn frame, by the app alone, and
  only while the tab shows; a look long after the last takes no peaks (nor
  the limiter); the window keeps drawing while it shows. A strip's meter
  is on its fader's ruler, so its marks read both.
  - `the_faders_are_set_on_the_mixer_and_again_on_a_new_one`
  - `a_meter_rises_at_once_and_falls_at_its_rate`
  - `strips_are_found_by_id_and_a_stale_look_takes_no_peaks`
  - `the_mix_tab_is_painted_from_its_cells`
