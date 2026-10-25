import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { LoaderCircleIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { describeError } from '@/lib/api/client';

/** Saves a clone voice through the app's existing /profiles route: there is no second voice store. */
export function CloneVoiceDialog({
  open,
  onOpenChange,
  onSave,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSave: (form: FormData) => Promise<string>;
}) {
  const { t } = useTranslation();
  const [name, setName] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [refText, setRefText] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const save = async () => {
    if (!name.trim() || !file || busy) return;
    setBusy(true);
    setError('');
    try {
      const form = new FormData();
      form.set('name', name.trim());
      form.set('kind', 'clone');
      form.set('ref_audio', file);
      if (refText.trim()) form.set('ref_text', refText.trim());
      await onSave(form);
      setName('');
      setFile(null);
      setRefText('');
      onOpenChange(false);
    } catch (reason) {
      setError(describeError(reason));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{t('qwenPlayground.clone_title')}</DialogTitle>
          <DialogDescription>{t('qwenPlayground.clone_desc')}</DialogDescription>
        </DialogHeader>
        <div className="space-y-3 text-sm">
          <label className="block space-y-1">
            <span className="text-xs text-muted-foreground">{t('qwenPlayground.clone_name')}</span>
            <Input value={name} onChange={(e) => setName(e.target.value)} />
          </label>
          <label className="block space-y-1">
            <span className="text-xs text-muted-foreground">{t('qwenPlayground.clone_file')}</span>
            <input
              type="file"
              accept="audio/*"
              className="block w-full text-xs"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </label>
          <label className="block space-y-1">
            <span className="text-xs text-muted-foreground">{t('qwenPlayground.clone_text')}</span>
            <textarea
              rows={2}
              value={refText}
              placeholder={t('qwenPlayground.clone_text_hint')}
              className="w-full resize-y rounded-md border border-input bg-input/20 p-2 text-sm"
              onChange={(e) => setRefText(e.target.value)}
            />
          </label>
          {error && (
            <p role="alert" className="text-xs text-destructive">
              {error}
            </p>
          )}
        </div>
        <DialogFooter>
          <DialogClose render={<Button variant="outline" />}>{t('qwenPlayground.cancel_dialog')}</DialogClose>
          <Button disabled={!name.trim() || !file || busy} onClick={() => void save()}>
            {busy && <LoaderCircleIcon className="animate-spin" />}
            {t('qwenPlayground.clone_save')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
