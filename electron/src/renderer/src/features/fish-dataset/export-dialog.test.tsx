import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) =>
      values ? `${key} ${JSON.stringify(values)}` : key,
  }),
}));

const player = vi.hoisted(() => ({
  stop: vi.fn(),
  toggle: vi.fn(),
  play: vi.fn(),
}));
vi.mock('./clip-player', () => ({
  useClipPlayer: () => ({ playing: null, play: player.play, toggle: player.toggle, stop: player.stop }),
}));

import { ExportDialog } from './export-dialog';
import {
  estimatePasses, keepTagsOf, parseDictionary, suggestSteps, trainingBody, EMPTY_TRAINING,
  qwenEstimate, qwenTrainBody, qwenTrainInvalid,
} from './export-options';
import type { ExportResult } from './use-fish-dataset';

afterEach(cleanup);

const OMNI = ['laughter', 'sigh'];
const TAGS: Array<[string, number]> = [
  ['slow', 30],
  ['laughing', 17],
  ['sighing', 33],
];

function setup(props: Partial<React.ComponentProps<typeof ExportDialog>> = {}, result?: Partial<ExportResult>) {
  const onExport = vi.fn(async (): Promise<ExportResult | null> => ({
    path: 'D:\\out\\set',
    clips: 3,
    hours: 0.1,
    speakers: {},
    formats: ['omnivoice'],
    ...result,
  }));
  render(
    <ExportDialog
      open
      onOpenChange={() => {}}
      clips={100}
      seconds={1800}
      excluded={0}
      knownTags={TAGS}
      omnivoiceTags={OMNI}
      onExport={onExport}
      onError={() => {}}
      {...props}
    />,
  );
  const dialog = screen.getByRole('dialog');
  fireEvent.change(within(dialog).getByLabelText('fishDataset.export_dest'), { target: { value: 'D:\\out' } });
  return { dialog, onExport };
}
const omniOn = (dialog: HTMLElement) => fireEvent.click(within(dialog).getByLabelText(/fishDataset\.format_omnivoice(?!_)/));
const exportButton = (dialog: HTMLElement) => within(dialog).getByRole('button', { name: /fishDataset\.export_button/ });
const sent = (onExport: ReturnType<typeof setup>['onExport']) => (onExport.mock.calls.at(-1) as unknown as [any])[0];

describe('tags table', () => {
  it('lists the selection tags only when the text is stripped, all set to Remove', () => {
    const { dialog } = setup();
    expect(within(dialog).queryByTestId('tag-row')).not.toBeInTheDocument();
    fireEvent.change(within(dialog).getByLabelText('fishDataset.export_text'), { target: { value: 'strip' } });
    const rows = within(dialog).getAllByTestId('tag-row');
    expect(rows).toHaveLength(3);
    expect(rows[1]).toHaveTextContent('[laughing]');
    expect(rows[1]).toHaveTextContent('17');
    expect(within(dialog).queryByLabelText('fishDataset.tags_filter')).not.toBeInTheDocument(); // few tags
  });

  it('says so, with no table, when the selection has no tags, and sends an empty list', async () => {
    const { dialog, onExport } = setup({ knownTags: [] });
    omniOn(dialog);
    expect(within(dialog).getByText('fishDataset.tags_none')).toBeInTheDocument();
    expect(within(dialog).queryByTestId('tag-row')).not.toBeInTheDocument();
    fireEvent.click(exportButton(dialog));
    await waitFor(() => expect(onExport).toHaveBeenCalled());
    expect(sent(onExport).keep_tags).toEqual([]);
  });

  it('sends one entry per kept or converted tag; Remove all and Keep all reset the rows', async () => {
    const { dialog, onExport } = setup();
    fireEvent.change(within(dialog).getByLabelText('fishDataset.export_text'), { target: { value: 'strip' } });
    fireEvent.change(within(dialog).getByLabelText(/tag_action .*"sighing"/), { target: { value: 'keep' } });
    fireEvent.change(within(dialog).getByLabelText(/tag_action .*"laughing"/), { target: { value: 'convert' } });
    fireEvent.change(within(dialog).getByLabelText(/tag_convert_to .*"laughing"/), { target: { value: ' (haha) ' } });
    fireEvent.click(exportButton(dialog));
    await waitFor(() => expect(onExport).toHaveBeenCalledTimes(1));
    expect(sent(onExport).keep_tags).toEqual(['[laughing]:(haha)', 'sighing']);

    fireEvent.click(within(dialog).getByRole('button', { name: 'fishDataset.tags_keep_all' }));
    fireEvent.click(exportButton(dialog));
    await waitFor(() => expect(onExport).toHaveBeenCalledTimes(2));
    expect(sent(onExport).keep_tags).toEqual(['slow', 'laughing', 'sighing']);

    fireEvent.click(within(dialog).getByRole('button', { name: 'fishDataset.tags_remove_all' }));
    fireEvent.click(exportButton(dialog));
    await waitFor(() => expect(onExport).toHaveBeenCalledTimes(3));
    expect(sent(onExport).keep_tags).toEqual([]);
  });

  it('blocks the export, with a reason, when more tags are kept than the backend accepts', () => {
    const many = Array.from({ length: 101 }, (_, i): [string, number] => [`tag${i}`, 1]);
    const { dialog } = setup({ knownTags: many, omnivoiceTags: [] });
    fireEvent.change(within(dialog).getByLabelText('fishDataset.export_text'), { target: { value: 'strip' } });
    expect(exportButton(dialog)).toBeEnabled(); // everything starts as Remove
    fireEvent.click(within(dialog).getByRole('button', { name: 'fishDataset.tags_keep_all' }));
    expect(exportButton(dialog)).toBeDisabled();
    expect(within(dialog).getByRole('alert')).toHaveTextContent('fishDataset.tags_too_many');
    fireEvent.click(within(dialog).getByRole('button', { name: 'fishDataset.tags_remove_all' }));
    expect(exportButton(dialog)).toBeEnabled();
  });

  it('treats an empty Convert as Remove', async () => {
    const { dialog, onExport } = setup();
    fireEvent.change(within(dialog).getByLabelText('fishDataset.export_text'), { target: { value: 'strip' } });
    fireEvent.change(within(dialog).getByLabelText(/tag_action .*"slow"/), { target: { value: 'convert' } });
    fireEvent.click(exportButton(dialog));
    await waitFor(() => expect(onExport).toHaveBeenCalled());
    expect(sent(onExport).keep_tags).toEqual([]);
  });

  it('shows a filter box when there are many tags', () => {
    const many: Array<[string, number]> = Array.from({ length: 14 }, (_, i) => [`t${i}`, 1]);
    const { dialog } = setup({ knownTags: many });
    omniOn(dialog);
    fireEvent.change(within(dialog).getByLabelText('fishDataset.tags_filter'), { target: { value: 't13' } });
    expect(within(dialog).getAllByTestId('tag-row')).toHaveLength(1);
  });

  it('fills the rows from a pasted dictionary and ignores tags that are not in the selection', async () => {
    const { dialog, onExport } = setup();
    omniOn(dialog);
    fireEvent.change(within(dialog).getByLabelText('fishDataset.tags_paste'), {
      target: { value: '[sighing]:(sad)\nlaughing=laughter\nghost=x' },
    });
    fireEvent.click(within(dialog).getByRole('button', { name: 'fishDataset.tags_paste_apply' }));
    fireEvent.click(within(dialog).getByLabelText('fishDataset.strict_tags')); // '(sad)' is not an OmniVoice tag
    fireEvent.click(exportButton(dialog));
    await waitFor(() => expect(onExport).toHaveBeenCalled());
    expect(sent(onExport).keep_tags).toEqual(['[laughing]:laughter', '[sighing]:(sad)']);
  });

  it('flags tags OmniVoice does not know and blocks the export until fixed or unticked', async () => {
    const { dialog, onExport } = setup();
    omniOn(dialog);
    expect(exportButton(dialog)).toBeEnabled();
    fireEvent.change(within(dialog).getByLabelText(/tag_action .*"sighing"/), { target: { value: 'convert' } });
    fireEvent.change(within(dialog).getByLabelText(/tag_convert_to .*"sighing"/), { target: { value: 'sad' } });
    expect(within(dialog).getByLabelText(/tag_convert_to .*"sighing"/)).toHaveClass('text-amber-500');
    expect(within(dialog).getByText(/fishDataset.tags_unknown/)).toBeInTheDocument();
    expect(exportButton(dialog)).toBeDisabled();

    fireEvent.change(within(dialog).getByLabelText(/tag_convert_to .*"sighing"/), { target: { value: '[Sigh]' } });
    expect(exportButton(dialog)).toBeEnabled();
    fireEvent.change(within(dialog).getByLabelText(/tag_convert_to .*"sighing"/), { target: { value: 'sad' } });
    fireEvent.click(within(dialog).getByLabelText('fishDataset.strict_tags'));
    expect(exportButton(dialog)).toBeEnabled();
    fireEvent.click(exportButton(dialog));
    await waitFor(() => expect(onExport).toHaveBeenCalled());
    expect(sent(onExport)).toMatchObject({ strict_tags: false, keep_tags: ['[sighing]:sad'] });
  });

  it('does not send strict_tags without the OmniVoice format', async () => {
    const { dialog, onExport } = setup();
    fireEvent.click(exportButton(dialog));
    await waitFor(() => expect(onExport).toHaveBeenCalled());
    expect(sent(onExport).strict_tags).toBeUndefined();
    expect(sent(onExport).training).toBeUndefined();
  });
});

describe('training setup', () => {
  it('appears only with the OmniVoice format and sends an empty object while nothing is changed', async () => {
    const { dialog, onExport } = setup();
    expect(within(dialog).queryByLabelText('fishDataset.train_write')).not.toBeInTheDocument();
    omniOn(dialog);
    expect(within(dialog).getByLabelText('fishDataset.train_write')).toBeChecked();
    fireEvent.click(exportButton(dialog));
    await waitFor(() => expect(onExport).toHaveBeenCalled());
    expect(sent(onExport).training).toEqual({});
  });

  it('sends no training object when the box is unticked', async () => {
    const { dialog, onExport } = setup();
    omniOn(dialog);
    fireEvent.click(within(dialog).getByLabelText('fishDataset.train_write'));
    fireEvent.click(exportButton(dialog));
    await waitFor(() => expect(onExport).toHaveBeenCalled());
    expect(sent(onExport).training).toBeUndefined();
  });

  it('sends only the fields that differ from the defaults', async () => {
    const { dialog, onExport } = setup();
    omniOn(dialog);
    fireEvent.change(within(dialog).getByLabelText('fishDataset.tr_steps'), { target: { value: '800' } });
    fireEvent.change(within(dialog).getByLabelText('fishDataset.tr_lora_r'), { target: { value: '16' } }); // the default
    fireEvent.change(within(dialog).getByLabelText('fishDataset.tr_mixed_precision'), { target: { value: 'fp16' } });
    fireEvent.change(within(dialog).getByLabelText('fishDataset.tr_seed'), { target: { value: '7' } });
    fireEvent.click(exportButton(dialog));
    await waitFor(() => expect(onExport).toHaveBeenCalled());
    expect(sent(onExport).training).toEqual({ steps: 800, mixed_precision: 'fp16', seed: 7 });
  });

  it('Full fine-tune hides the LoRA fields and moves the learning rate default unless it was edited', async () => {
    const { dialog, onExport } = setup();
    omniOn(dialog);
    fireEvent.change(within(dialog).getByLabelText('fishDataset.tr_mode'), { target: { value: 'full' } });
    expect(within(dialog).queryByLabelText('fishDataset.tr_lora_r')).not.toBeInTheDocument();
    expect(within(dialog).queryByLabelText('fishDataset.tr_train_heads')).not.toBeInTheDocument();
    expect(within(dialog).getByLabelText('fishDataset.tr_learning_rate')).toHaveValue(5e-5);
    fireEvent.click(exportButton(dialog));
    await waitFor(() => expect(onExport).toHaveBeenCalledTimes(1));
    expect(sent(onExport).training).toEqual({ mode: 'full' });

    // an edited rate is kept, even if it equals the LoRA default
    fireEvent.change(within(dialog).getByLabelText('fishDataset.tr_learning_rate'), { target: { value: '0.0001' } });
    fireEvent.click(exportButton(dialog));
    await waitFor(() => expect(onExport).toHaveBeenCalledTimes(2));
    expect(sent(onExport).training).toEqual({ mode: 'full', learning_rate: 0.0001 });
  });

  it('sets steps from passes over the data, using the training share of the audio', () => {
    const { dialog } = setup({ seconds: 1800 }); // 5% validation by default -> 1710 s
    omniOn(dialog);
    fireEvent.change(within(dialog).getByLabelText('fishDataset.tr_passes_set'), { target: { value: '10' } });
    const steps = suggestSteps(1710, 4096, 10);
    expect(steps).toBe(121);
    expect(within(dialog).getByLabelText('fishDataset.tr_steps')).toHaveValue(steps);
  });

  it('shows the passes estimate and warns about overfitting under an hour of audio', () => {
    const { dialog } = setup({ seconds: 600 });
    omniOn(dialog);
    const estimate = estimatePasses(570, 2000, 4096);
    expect(estimate).toBeGreaterThan(40);
    expect(within(dialog).getByTestId('passes-estimate')).toHaveTextContent(`"passes":${estimate}`);
    expect(within(dialog).getByText(/fishDataset\.tr_overfit/)).toBeInTheDocument();
  });

  it('does not warn about overfitting with an hour or more of audio', () => {
    const { dialog } = setup({ seconds: 7200 });
    omniOn(dialog);
    fireEvent.change(within(dialog).getByLabelText('fishDataset.tr_steps'), { target: { value: '5000' } });
    expect(within(dialog).queryByText(/fishDataset\.tr_overfit/)).not.toBeInTheDocument();
  });

  it('blocks the export while a field is outside the range the backend accepts', () => {
    const { dialog } = setup();
    omniOn(dialog);
    fireEvent.change(within(dialog).getByLabelText('fishDataset.tr_steps'), { target: { value: '5' } });
    expect(exportButton(dialog)).toBeDisabled();
    fireEvent.change(within(dialog).getByLabelText('fishDataset.tr_steps'), { target: { value: '500' } });
    expect(exportButton(dialog)).toBeEnabled();
  });

  it('reports the written training files and the estimate after the export', async () => {
    const { dialog } = setup(
      {},
      {
        training: {
          files: ['train_config.json', 'data_config.json', 'train.sh', 'COLAB.md'],
          estimate: { passes: 11.2, steps_per_pass: 178, note: 'estimate' },
        },
      },
    );
    omniOn(dialog);
    fireEvent.click(exportButton(dialog));
    const status = await within(dialog).findByRole('status');
    expect(status).toHaveTextContent('train_config.json, data_config.json, train.sh, COLAB.md');
    expect(status).toHaveTextContent('"passes":11.2');
    expect(within(status).getByRole('button', { name: /fishStory\.reveal/ })).toBeInTheDocument();
  });
});

describe('option helpers', () => {
  it('builds keep_tags entries and parses the dictionary syntax', () => {
    const tags = ['a', 'b'];
    expect(keepTagsOf(tags, { a: { mode: 'keep', to: '' }, b: { mode: 'convert', to: ' x ' } })).toEqual(['a', '[b]:x']);
    expect(parseDictionary('[a]:(sad); b=b', tags)).toEqual({
      a: { mode: 'convert', to: '(sad)' },
      b: { mode: 'keep', to: '' },
    });
    expect(trainingBody(EMPTY_TRAINING)).toEqual({});
  });
});

const qwenOn = (dialog: HTMLElement) => fireEvent.click(within(dialog).getByLabelText(/fishDataset\.format_qwen3_tts(?!_)/));
const QWEN_RESULT: Partial<ExportResult> = {
  formats: ['qwen3_tts'],
  qwen3: {
    accepted: 90,
    rejected: 3,
    reference: { source: 'c/run/001.wav' },
    warnings: [{ code: 'clipping', message: 'two clips touch full scale' }],
    zip: { path: 'D:\\out\\set.zip', size: 10 },
    files: ['train.jsonl', 'training_settings.json', 'colab_setup.sh'],
    commands: {
      notebook: ['!bash /content/set/colab_setup.sh'],
      colab_cli: ['colab new -s qwen-train --gpu A100'],
      official: { prepare: 'python prepare_data.py', train: 'python sft_12hz.py --num_epochs 7', checkpoint_dir: '/content/drive/MyDrive/X/Y/checkpoints' },
    },
  },
};

describe('Qwen3-TTS in the shared Training setup', () => {
  it('switches the Training setup fields by format and replaces the other formats', () => {
    const { dialog } = setup();
    omniOn(dialog);
    expect(within(dialog).getByLabelText('fishDataset.tr_steps')).toBeInTheDocument();
    expect(within(dialog).queryByLabelText('fishDataset.qtr_epochs')).not.toBeInTheDocument();
    omniOn(dialog); // untick
    qwenOn(dialog);
    expect(within(dialog).getByLabelText('fishDataset.qtr_epochs')).toHaveValue(3);
    expect(within(dialog).getByLabelText('fishDataset.qtr_learning_rate')).toHaveValue(2e-5);
    expect(within(dialog).getByLabelText('fishDataset.qtr_batch_size')).toHaveValue(2);
    expect(within(dialog).queryByLabelText('fishDataset.tr_steps')).not.toBeInTheDocument();
    expect(within(dialog).getByText('fishDataset.qtr_colab_only')).toBeInTheDocument();
    expect(within(dialog).queryByLabelText('fishDataset.export_val')).not.toBeInTheDocument(); // no validation split
    expect(within(dialog).queryByTestId('tag-row')).not.toBeInTheDocument(); // tags are always removed
    expect(within(dialog).getByLabelText(/fishDataset\.format_jsonl(?!_)/)).toBeDisabled();
    expect(within(dialog).getByTestId('qwen-checkpoint-path')).toHaveTextContent('My Drive/VoiceStudio-training/<speaker>/checkpoints');
  });

  it('sends only changed training fields and the Qwen3-TTS options', async () => {
    const { dialog, onExport } = setup({}, QWEN_RESULT);
    qwenOn(dialog);
    fireEvent.click(exportButton(dialog));
    await waitFor(() => expect(onExport).toHaveBeenCalledTimes(1));
    expect(sent(onExport)).toMatchObject({ formats: ['qwen3_tts'], val_fraction: 0, sample_rate: 24000, keep_tags: [], training: {} });
    expect(sent(onExport).qwen3).toMatchObject({ speaker_name: '', reference_key: null, make_zip: true, min_duration: 0.8, max_duration: 30 });

    fireEvent.change(within(dialog).getByLabelText('fishDataset.qtr_epochs'), { target: { value: '7' } });
    fireEvent.change(within(dialog).getByLabelText('fishDataset.qtr_learning_rate'), { target: { value: '0.00001' } });
    fireEvent.change(within(dialog).getByLabelText('fishDataset.qtr_drive_folder'), { target: { value: 'X/Y' } });
    fireEvent.change(within(dialog).getByLabelText('fishDataset.qwen_speaker'), { target: { value: 'Sunny' } });
    expect(within(dialog).getByTestId('qwen-checkpoint-path')).toHaveTextContent('My Drive/X/Y/Sunny/checkpoints');
    fireEvent.click(exportButton(dialog));
    await waitFor(() => expect(onExport).toHaveBeenCalledTimes(2));
    expect(sent(onExport).training).toEqual({ epochs: 7, learning_rate: 0.00001, drive_folder: 'X/Y' });
    expect(sent(onExport).qwen3.speaker_name).toBe('Sunny');
  });

  it('shows an estimate from the script loop, not an audio rate', () => {
    const { dialog } = setup({ clips: 100 });
    qwenOn(dialog);
    expect(within(dialog).getByTestId('qwen-estimate')).toHaveTextContent('"batches":50,"updates":13,"total":39');
    fireEvent.change(within(dialog).getByLabelText('fishDataset.qtr_batch_size'), { target: { value: '4' } });
    expect(within(dialog).getByTestId('qwen-estimate')).toHaveTextContent('"batches":25,"updates":7,"total":21');
    expect(within(dialog).getByText('fishDataset.qtr_estimate_note')).toBeInTheDocument();
    expect(within(dialog).queryByTestId('passes-estimate')).not.toBeInTheDocument();
  });

  it('blocks the export for out-of-range values, a bad folder, or several speakers', () => {
    const { dialog } = setup();
    qwenOn(dialog);
    expect(exportButton(dialog)).toBeEnabled();
    fireEvent.change(within(dialog).getByLabelText('fishDataset.qtr_epochs'), { target: { value: '0' } });
    expect(exportButton(dialog)).toBeDisabled();
    fireEvent.change(within(dialog).getByLabelText('fishDataset.qtr_epochs'), { target: { value: '2.5' } });
    expect(exportButton(dialog)).toBeDisabled();
    fireEvent.change(within(dialog).getByLabelText('fishDataset.qtr_epochs'), { target: { value: '3' } });
    fireEvent.change(within(dialog).getByLabelText('fishDataset.qtr_drive_folder'), { target: { value: '../x' } });
    expect(exportButton(dialog)).toBeDisabled();
    fireEvent.change(within(dialog).getByLabelText('fishDataset.qtr_drive_folder'), { target: { value: 'ok' } });
    expect(exportButton(dialog)).toBeEnabled();
    cleanup();
    const again = setup({ speakerCount: 3 });
    qwenOn(again.dialog);
    expect(within(again.dialog).getByRole('alert')).toHaveTextContent('qwen_multi');
    expect(exportButton(again.dialog)).toBeDisabled();
  });

  it('lists the written files, the start commands and the Drive checkpoint folder afterwards', async () => {
    const { dialog } = setup({}, QWEN_RESULT);
    qwenOn(dialog);
    fireEvent.click(exportButton(dialog));
    const status = await within(dialog).findByRole('status');
    expect(status).toHaveTextContent('train.jsonl, training_settings.json, colab_setup.sh');
    expect(within(status).getByTestId('qwen-start-commands')).toHaveTextContent('!bash /content/set/colab_setup.sh');
    expect(within(status).getByTestId('qwen-start-commands')).toHaveTextContent('colab new -s qwen-train');
    expect(status).toHaveTextContent('/content/drive/MyDrive/X/Y/checkpoints');
    expect(status).toHaveTextContent('--num_epochs 7');
    expect(status).toHaveTextContent('two clips touch full scale');
    expect(within(status).queryByRole('button', { name: /fishTraining\.train_open/ })).not.toBeInTheDocument(); // no local runner
  });
});

describe('Qwen option helpers', () => {
  it('sends only values that differ from the official script defaults', () => {
    expect(qwenTrainBody({ edits: {} })).toEqual({});
    expect(qwenTrainBody({ edits: { epochs: '3', batch_size: '8', base_model: 'Qwen/Qwen3-TTS-12Hz-0.6B-Base' } })).toEqual({
      batch_size: 8,
      base_model: 'Qwen/Qwen3-TTS-12Hz-0.6B-Base',
    });
    expect(qwenTrainInvalid({ edits: { base_model: 'x; ls' } })).toEqual(['base_model']);
    expect(qwenEstimate(0, 2, 3)).toBeNull();
  });
});

const LEVELS = {
  measured: 10,
  total: 10,
  min: -26,
  median: -25,
  max: -14,
  counts: [[-26, 4], [-25, 4], [-14, 2]] as Array<[number, number]>,
};

describe('match loudness', () => {
  it('is ticked with -22 dB for the training formats and sends it without the user touching the field', async () => {
    const { dialog, onExport } = setup({}, QWEN_RESULT);
    qwenOn(dialog);
    expect(within(dialog).getByLabelText('fishDataset.export_loudness')).toBeChecked();
    expect(within(dialog).getByLabelText('fishDataset.export_loudness_db')).toHaveValue(-22);
    fireEvent.click(exportButton(dialog));
    await waitFor(() => expect(onExport).toHaveBeenCalledTimes(1));
    expect(sent(onExport).loudness_db).toBe(-22);

    cleanup();
    const omni = setup();
    omniOn(omni.dialog);
    fireEvent.click(exportButton(omni.dialog));
    await waitFor(() => expect(omni.onExport).toHaveBeenCalled());
    expect(sent(omni.onExport).loudness_db).toBe(-22);
  });

  it('stays off for the other formats, and the user can untick it for the training formats', async () => {
    const plain = setup();
    expect(within(plain.dialog).getByLabelText('fishDataset.export_loudness')).not.toBeChecked();
    fireEvent.click(exportButton(plain.dialog));
    await waitFor(() => expect(plain.onExport).toHaveBeenCalled());
    expect(sent(plain.onExport).loudness_db).toBeNull();
    cleanup();

    const { dialog, onExport } = setup({}, QWEN_RESULT);
    qwenOn(dialog);
    fireEvent.click(within(dialog).getByLabelText('fishDataset.export_loudness'));
    expect(within(dialog).queryByLabelText('fishDataset.export_loudness_db')).not.toBeInTheDocument();
    fireEvent.click(exportButton(dialog));
    await waitFor(() => expect(onExport).toHaveBeenCalled());
    expect(sent(onExport).loudness_db).toBeNull();
  });

  it('offers -19, -22 and -23 presets, accepts a typed value and refuses one outside -60..-3', async () => {
    const { dialog, onExport } = setup({}, QWEN_RESULT);
    qwenOn(dialog);
    fireEvent.click(within(dialog).getByRole('button', { name: /fishDataset\.level_preset .*-19/ }));
    expect(within(dialog).getByLabelText('fishDataset.export_loudness_db')).toHaveValue(-19);
    fireEvent.click(within(dialog).getByRole('button', { name: /fishDataset\.level_preset .*-23/ }));
    fireEvent.click(exportButton(dialog));
    await waitFor(() => expect(onExport).toHaveBeenCalledTimes(1));
    expect(sent(onExport).loudness_db).toBe(-23);

    fireEvent.change(within(dialog).getByLabelText('fishDataset.export_loudness_db'), { target: { value: '-2' } });
    expect(exportButton(dialog)).toBeDisabled();
    expect(within(dialog).getByText(/level_invalid/)).toBeInTheDocument();
    fireEvent.change(within(dialog).getByLabelText('fishDataset.export_loudness_db'), { target: { value: '-61' } });
    expect(exportButton(dialog)).toBeDisabled();
    fireEvent.change(within(dialog).getByLabelText('fishDataset.export_loudness_db'), { target: { value: '-30.5' } });
    expect(exportButton(dialog)).toBeEnabled();
    expect(within(dialog).getByText('fishDataset.export_loudness_hint')).toBeInTheDocument();
  });

  it('shows the selection levels and how many clips sit below and above the target', () => {
    const { dialog } = setup({ levels: LEVELS });
    omniOn(dialog);
    const summary = within(dialog).getByTestId('level-summary');
    expect(summary).toHaveTextContent('"min":-26,"max":-14,"median":-25');
    expect(summary).toHaveTextContent('"below":8,"above":2,"within":0,"target":-22');
    fireEvent.change(within(dialog).getByLabelText('fishDataset.export_loudness_db'), { target: { value: '-15' } });
    expect(within(dialog).getByTestId('level-summary')).toHaveTextContent('"below":8,"above":0,"within":2,"target":-15');
  });

  it('says clips are not measured and offers the existing Measure audio action', () => {
    const onMeasure = vi.fn();
    const { dialog } = setup({ levels: { ...LEVELS, measured: 0, min: null, max: null, median: null, counts: [] }, onMeasure });
    expect(within(dialog).getByText('fishDataset.level_unmeasured')).toBeInTheDocument();
    fireEvent.click(within(dialog).getByRole('button', { name: 'fishDataset.analyze' }));
    expect(onMeasure).toHaveBeenCalled();
  });

  it('reports per chapter what matching did, including clips the peak cap stopped', async () => {
    const { dialog } = setup({}, {
      loudness: {
        target_rms_db: -22, clips: 40, adjusted: 38, peak_limited: 3, worst_shortfall_db: 4.2,
        chapters: {
          '3193_hot': { clips: 10, adjusted: 10, peak_limited: 3, worst_shortfall_db: 4.2, median_before_db: -16.1, median_after_db: -22 },
        },
      },
    });
    omniOn(dialog);
    fireEvent.click(exportButton(dialog));
    const result = await within(dialog).findByTestId('loudness-result');
    expect(result).toHaveTextContent('"target":-22,"adjusted":38,"clips":40');
    expect(result).toHaveTextContent('"chapter":"3193_hot","before":-16.1,"after":-22');
    expect(result).toHaveTextContent('"count":3,"db":4.2');
  });
});

const CLIPS = [
  { key: 'k1', label: '3001 #1', voice_id: 'v-narr', voice_name: 'Narrator', text: 'Rain fell on the roof.', duration_s: 8.2, stats: null },
  { key: 'k2', label: '3002 #4', voice_id: 'v-narr', voice_name: 'Narrator', text: 'A calm sentence.', duration_s: 9.5, stats: null },
  { key: 'k9', label: '3002 #9', voice_id: 'v-noct', voice_name: 'Noctis', text: 'Another voice.', duration_s: 5, stats: null },
];
const CANDIDATE = (key: string, index: number) => ({
  key,
  chapter: '3001',
  index,
  text: `Candidate ${index}.`,
  duration_s: 9,
  stats: { peak: 0.5, rms_db: -22.5, clip: 0, lead_ms: 100, tail_ms: 200, pauses: 1, max_gap_ms: 300, noise_floor_db: -50 },
  score: -0.1 * index,
  reasons: ['not_clipped', 'neutral'],
  cautions: index === 2 ? ['many_pauses'] : [],
  relaxed: null,
});
const SUGGESTION = {
  speaker: { voice_id: 'v-narr', name: 'Narrator' },
  considered: 6,
  unmeasured: 0,
  clips: 6,
  candidates: [1, 2, 3, 4, 5].map((i) => CANDIDATE(`c${i}`, i)),
};

describe('reference clip panel', () => {
  const panel = (props: Partial<React.ComponentProps<typeof ExportDialog>> = {}, result?: Partial<ExportResult>) => {
    const onSuggestReferences = vi.fn(async () => SUGGESTION);
    const onReferenceChange = vi.fn();
    const made = setup({ shortlist: CLIPS, speakerId: 'v-narr', onSuggestReferences, onReferenceChange, ...props }, result ?? QWEN_RESULT);
    qwenOn(made.dialog);
    return { ...made, onSuggestReferences, onReferenceChange };
  };

  it('says it is chosen automatically and shows the clip that would be picked and why', async () => {
    const { dialog, onSuggestReferences } = panel();
    const current = within(dialog).getByTestId('reference-current');
    await waitFor(() => expect(current).toHaveTextContent('Candidate 1.'));
    expect(current).toHaveTextContent('fishDataset.ref_auto_now');
    expect(current).toHaveTextContent('fishDataset.ref_reason_not_clipped');
    expect(onSuggestReferences).toHaveBeenCalledWith(5, 0.8, 30);
  });

  it('lists the top 5 with reasons, and Use this picks one', async () => {
    const { dialog, onReferenceChange } = panel();
    const rows = await within(dialog).findAllByTestId('reference-suggestion');
    expect(rows).toHaveLength(5);
    expect(rows[1]).toHaveTextContent('fishDataset.ref_caution_many_pauses');
    expect(rows[0]).toHaveTextContent('fishDataset.ref_reason_neutral');
    fireEvent.click(within(rows[2]).getByRole('button', { name: /fishDataset\.ref_use_clip/ }));
    expect(onReferenceChange).toHaveBeenCalledWith(expect.objectContaining({ key: 'c3', label: '3001 #3' }));
  });

  it('Suggest asks again for the current selection', async () => {
    const { dialog, onSuggestReferences } = panel();
    await within(dialog).findAllByTestId('reference-suggestion');
    fireEvent.click(within(dialog).getByRole('button', { name: /fishDataset\.ref_suggest$/ }));
    await waitFor(() => expect(onSuggestReferences).toHaveBeenCalledTimes(2));
  });

  it('lists only the shortlisted clips of the exported speaker, one radio each, plus Automatic', async () => {
    const onRemove = vi.fn();
    const { dialog, onReferenceChange } = panel({ onRemoveShortlist: onRemove });
    const choices = within(dialog).getByTestId('reference-choices');
    expect(within(choices).getAllByRole('radio')).toHaveLength(3); // Automatic + two Narrator clips
    expect(choices).toHaveTextContent('A calm sentence.');
    expect(choices).not.toHaveTextContent('Another voice.');
    expect(within(choices).getByRole('alert')).toHaveTextContent('"count":1');
    fireEvent.click(within(choices).getByRole('button', { name: 'fishDataset.ref_remove_other' }));
    expect(onRemove).toHaveBeenCalledWith(['k9']);
    fireEvent.click(within(choices).getByLabelText(/fishDataset\.ref_choose_clip .*3002 #4/));
    expect(onReferenceChange).toHaveBeenCalledWith(expect.objectContaining({ key: 'k2' }));
    await within(dialog).findAllByTestId('reference-suggestion');
  });

  it('shows the chosen reference with its details, Automatic clears it, and exactly its key is sent', async () => {
    const chosen = { key: 'k2', label: '3002 #4', text: 'A calm sentence.', duration_s: 9.5, stats: null };
    const { dialog, onExport, onReferenceChange } = panel({ reference: chosen });
    expect(within(dialog).getByTestId('reference-current')).toHaveTextContent('"label":"3002 #4"');
    expect(within(dialog).getByTestId('reference-current')).toHaveTextContent('A calm sentence.');
    fireEvent.click(within(dialog).getByLabelText('fishDataset.ref_automatic'));
    expect(onReferenceChange).toHaveBeenCalledWith(null);
    fireEvent.click(exportButton(dialog));
    await waitFor(() => expect(onExport).toHaveBeenCalled());
    expect(sent(onExport).qwen3.reference_key).toBe('k2');
    await within(dialog).findAllByTestId('reference-suggestion');
  });

  it('sends a null reference when Automatic is chosen', async () => {
    const { dialog, onExport } = panel();
    fireEvent.click(exportButton(dialog));
    await waitFor(() => expect(onExport).toHaveBeenCalled());
    expect(sent(onExport).qwen3.reference_key).toBeNull();
    await within(dialog).findAllByTestId('reference-suggestion');
  });

  it('says when clips are not measured and offers to measure first', async () => {
    const onMeasure = vi.fn();
    const none = { ...SUGGESTION, unmeasured: 6, candidates: [] };
    const { dialog } = panel({ onSuggestReferences: vi.fn(async () => none), onMeasure });
    const note = await within(dialog).findByTestId('reference-unmeasured');
    expect(note).toHaveTextContent('"unmeasured":6,"clips":6');
    fireEvent.click(within(note).getByRole('button', { name: 'fishDataset.analyze' }));
    expect(onMeasure).toHaveBeenCalled();
    expect(within(dialog).queryByTestId('reference-suggestion')).not.toBeInTheDocument();
  });

  it('shows the backend message when the selection cannot be ranked', async () => {
    const { dialog } = panel({ onSuggestReferences: vi.fn(async () => { throw new Error('Pick one speaker'); }) });
    expect(await within(dialog).findByText(/Pick one speaker/)).toBeInTheDocument();
  });

  it('plays a clip with the shared player, and stops playback when the dialog closes', async () => {
    player.stop.mockClear();
    const view = render(
      <ExportDialog open onOpenChange={() => {}} clips={5} seconds={50} excluded={0} knownTags={[]} onExport={vi.fn()}
        onError={() => {}} shortlist={CLIPS} speakerId="v-narr" onSuggestReferences={async () => SUGGESTION} />,
    );
    const dialog = screen.getByRole('dialog');
    qwenOn(dialog);
    fireEvent.click(within(dialog).getAllByRole('button', { name: /fishDataset\.ref_play .*3001 #1/ })[0]);
    expect(player.toggle).toHaveBeenCalledWith('k1');
    await within(dialog).findAllByTestId('reference-suggestion');
    player.stop.mockClear();
    view.rerender(
      <ExportDialog open={false} onOpenChange={() => {}} clips={5} seconds={50} excluded={0} knownTags={[]} onExport={vi.fn()}
        onError={() => {}} shortlist={CLIPS} speakerId="v-narr" onSuggestReferences={async () => SUGGESTION} />,
    );
    expect(player.stop).toHaveBeenCalled();
  });
});
