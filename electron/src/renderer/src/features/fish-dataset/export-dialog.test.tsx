import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) =>
      values ? `${key} ${JSON.stringify(values)}` : key,
  }),
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

  it('sends the starred reference clip, and clears it', async () => {
    const onClear = vi.fn();
    const { dialog, onExport } = setup({ referenceKey: 'run:c1', referenceLabel: '3001 #4', onClearReference: onClear }, QWEN_RESULT);
    qwenOn(dialog);
    expect(within(dialog).getByText(/qwen_ref_clip .*3001 #4/)).toBeInTheDocument();
    fireEvent.click(within(dialog).getByRole('button', { name: 'fishDataset.qwen_ref_clear' }));
    expect(onClear).toHaveBeenCalled();
    fireEvent.click(exportButton(dialog));
    await waitFor(() => expect(onExport).toHaveBeenCalled());
    expect(sent(onExport).qwen3.reference_key).toBe('run:c1');
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
