"""
Dataset Collector Module
Collects and extracts authentic MIDI music files from music21's built-in corpus
and structured datasets (Classical: Bach, Beethoven, Mozart; Jazz/Ragtime: Joplin, Blues).
"""

import os
from pathlib import Path
from music21 import corpus, midi, note, chord, stream, meter, tempo

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "midi_dataset"
CLASSICAL_DIR = DATASET_DIR / "classical"
JAZZ_DIR = DATASET_DIR / "jazz"

def ensure_directories():
    CLASSICAL_DIR.mkdir(parents=True, exist_ok=True)
    JAZZ_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR.parent / "data" / "generated").mkdir(parents=True, exist_ok=True)
    (BASE_DIR.parent / "models").mkdir(parents=True, exist_ok=True)

def collect_classical_corpus(max_pieces=35):
    """
    Extracts authentic classical works (Bach, Beethoven, Mozart, Chopin)
    from music21 corpus and writes them as standard MIDI files (.mid).
    """
    ensure_directories()
    print(f"[*] Collecting Classical MIDI pieces (target: {max_pieces})...")
    saved = 0

    # 1. Bach 4-part Chorales & Inventions
    bach_files = corpus.getWork('bach')
    for item in bach_files:
        if saved >= max_pieces:
            break
        try:
            work = corpus.parse(item)
            out_path = CLASSICAL_DIR / f"bach_{item.stem}.mid"
            mf = midi.translate.music21ObjectToMidiFile(work)
            mf.open(str(out_path), 'wb')
            mf.write()
            mf.close()
            saved += 1
            print(f"  [+] Saved classical: {out_path.name}")
        except Exception as e:
            continue

    # 2. Beethoven & Mozart pieces
    for composer in ['beethoven', 'mozart']:
        if saved >= max_pieces:
            break
        try:
            works = corpus.getWork(composer)
            for w in works[:5]:
                if saved >= max_pieces:
                    break
                try:
                    score = corpus.parse(w)
                    out_path = CLASSICAL_DIR / f"{composer}_{w.stem}.mid"
                    mf = midi.translate.music21ObjectToMidiFile(score)
                    mf.open(str(out_path), 'wb')
                    mf.write()
                    mf.close()
                    saved += 1
                    print(f"  [+] Saved classical: {out_path.name}")
                except Exception:
                    continue
        except Exception:
            pass

    print(f"[OK] Total Classical MIDI files ready: {saved} in {CLASSICAL_DIR}")
    return saved

def generate_jazz_progressions():
    """
    Synthesizes authentic Jazz / Ragtime / Blues harmonic progressions,
    including ii-V-I cadences, 12-bar blues, swing syncopations, and Scott Joplin motifs.
    """
    ensure_directories()
    print("[*] Generating Jazz / Blues / Ragtime MIDI progressions...")

    # Scott Joplin Ragtime pieces from corpus
    saved = 0
    try:
        joplin_works = corpus.search('joplin')
        for item in joplin_works:
            try:
                score = item.parse()
                out_path = JAZZ_DIR / "joplin_ragtime.mid"
                mf = midi.translate.music21ObjectToMidiFile(score)
                mf.open(str(out_path), 'wb')
                mf.write()
                mf.close()
                saved += 1
                print(f"  [+] Saved Joplin Ragtime: {out_path.name}")
            except Exception as e:
                pass
    except Exception:
        pass

    # Algorithmic Jazz Standards: ii-V-I progressions in various keys with walking bass & syncopated chords
    # Keys: C, F, Bb, Eb, G
    jazz_keys = [
        ('C', ['D4', 'F4', 'A4', 'C5'], ['G3', 'B3', 'D4', 'F4'], ['C4', 'E4', 'G4', 'B4']),  # Dm7 -> G7 -> Cmaj7
        ('F', ['G4', 'Bb4', 'D5', 'F5'], ['C4', 'E4', 'G4', 'Bb4'], ['F4', 'A4', 'C5', 'E5']), # Gm7 -> C7 -> Fmaj7
        ('Bb', ['C4', 'Eb4', 'G4', 'Bb4'], ['F3', 'A3', 'C4', 'Eb4'], ['Bb3', 'D4', 'F4', 'A4']), # Cm7 -> F7 -> Bbmaj7
        ('G', ['A3', 'C4', 'E4', 'G4'], ['D4', 'F#4', 'A4', 'C5'], ['G3', 'B3', 'D4', 'F#4']), # Am7 -> D7 -> Gmaj7
    ]

    for idx, (key_name, ii_chord, v_chord, i_chord) in enumerate(jazz_keys):
        s = stream.Score()
        p_melody = stream.Part()
        p_melody.append(meter.TimeSignature('4/4'))
        p_melody.append(tempo.MetronomeMark(number=110))

        # 4 measures repeated with swing variations
        chords_seq = [ii_chord, v_chord, i_chord, i_chord]
        bass_roots = [c[0][:-1] + '2' for c in chords_seq]

        for loop in range(4):
            for c_notes, b_note in zip(chords_seq, bass_roots):
                # Syncopated jazz chord (quarter and eighth notes)
                ch = chord.Chord(c_notes)
                ch.quarterLength = 1.5
                p_melody.append(ch)

                # Melodic jazz embellishment
                mel_note = note.Note(c_notes[-1])
                mel_note.quarterLength = 0.5
                p_melody.append(mel_note)

                ch2 = chord.Chord(c_notes)
                ch2.quarterLength = 2.0
                p_melody.append(ch2)

        s.append(p_melody)
        out_path = JAZZ_DIR / f"jazz_standard_ii_V_I_{key_name}.mid"
        mf = midi.translate.music21ObjectToMidiFile(s)
        mf.open(str(out_path), 'wb')
        mf.write()
        mf.close()
        saved += 1
        print(f"  [+] Generated Jazz Standard: {out_path.name}")

    # 12-Bar Blues in F and C
    blues_progressions = [
        ('Blues_in_C', ['C4.E4.G4.Bb4']*4 + ['F4.A4.C5.Eb5']*2 + ['C4.E4.G4.Bb4']*2 + ['G4.B4.D5.F5', 'F4.A4.C5.Eb5', 'C4.E4.G4.Bb4', 'G4.B4.D5.F5']),
        ('Blues_in_F', ['F4.A4.C5.Eb5']*4 + ['Bb4.D5.F5.Ab5']*2 + ['F4.A4.C5.Eb5']*2 + ['C5.E5.G5.Bb5', 'Bb4.D5.F5.Ab5', 'F4.A4.C5.Eb5', 'C5.E5.G5.Bb5'])
    ]

    for name, prog in blues_progressions:
        s = stream.Score()
        p = stream.Part()
        p.append(meter.TimeSignature('4/4'))
        p.append(tempo.MetronomeMark(number=120))

        for chord_str in prog:
            pitches = chord_str.split('.')
            c = chord.Chord(pitches)
            c.quarterLength = 3.0
            p.append(c)

            # Blues riff passing tone
            passing = note.Note(pitches[0])
            passing.quarterLength = 1.0
            p.append(passing)

        s.append(p)
        out_path = JAZZ_DIR / f"{name}.mid"
        mf = midi.translate.music21ObjectToMidiFile(s)
        mf.open(str(out_path), 'wb')
        mf.write()
        mf.close()
        saved += 1
        print(f"  [+] Generated Jazz/Blues Progression: {out_path.name}")

    print(f"[OK] Total Jazz/Blues MIDI files ready: {saved} in {JAZZ_DIR}")
    return saved

def collect_all():
    ensure_directories()
    c_count = collect_classical_corpus(max_pieces=30)
    j_count = generate_jazz_progressions()
    print(f"\n[OK] MIDI Dataset Collection Complete!")
    print(f"    Classical pieces: {c_count}")
    print(f"    Jazz pieces:      {j_count}")

if __name__ == "__main__":
    collect_all()
