import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) =>
      values ? `${key} ${JSON.stringify(values)}` : key,
  }),
}));

import { ExportDialog } from './export-dialog';
import { estimatePasses, keepTagsOf, parseDictionary, suggestSteps, trainingBody, EMPTY_TRAINING } from './export-options';
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
