<script lang="ts">
	import * as actions from '../actions';
	import { isValidGgufFile } from '@wllama/wllama';
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import * as Dialog from '@ggml-org/llama-ui/ui/dialog';
	import { Input } from '@ggml-org/llama-ui/ui/input';
	import { Label } from '@ggml-org/llama-ui/ui/label';
	import * as Select from '@ggml-org/llama-ui/ui/select';
	import { LoaderCircle } from '@lucide/svelte';

	let { open = $bindable(false) }: { open: boolean } = $props();

	let hfRepo = $state('');
	let hfFile = $state('');
	let hfMmprojFile = $state('');
	let modelFiles = $state<string[]>([]);
	let mmprojFiles = $state<string[]>([]);
	let err = $state('');
	let adding = $state(false);

	// list the GGUF files of the repo, 500ms after the last key press
	$effect(() => {
		const repo = hfRepo.trim();
		modelFiles = [];
		mmprojFiles = [];
		hfFile = '';
		hfMmprojFile = '';
		err = '';
		if (repo.length < 2) return;
		const abort = new AbortController();
		const timer = setTimeout(async () => {
			try {
				const res = await fetch(`https://huggingface.co/api/models/${repo}`, {
					signal: abort.signal
				});
				const data: { siblings?: { rfilename: string }[] } = await res.json();
				if (!data.siblings) {
					err = 'No model found, or it is private';
					return;
				}
				const files = data.siblings.map((s) => s.rfilename).filter((f) => isValidGgufFile(f));
				modelFiles = files.filter((f) => !f.toLowerCase().includes('mmproj'));
				mmprojFiles = files.filter((f) => f.toLowerCase().includes('mmproj'));
			} catch (e) {
				if ((e as Error).name !== 'AbortError') err = (e as Error).message ?? 'Unknown error';
			}
		}, 500);
		return () => {
			clearTimeout(timer);
			abort.abort();
		};
	});

	async function onSubmit() {
		const base = `https://huggingface.co/${hfRepo.trim()}/resolve/main`;
		adding = true;
		err = '';
		try {
			await actions.addCustomModel(
				`${base}/${hfFile}`,
				hfMmprojFile ? `${base}/${hfMmprojFile}` : undefined
			);
			hfRepo = '';
			open = false;
		} catch (e) {
			err = (e as Error).message ?? 'Unknown error';
		} finally {
			adding = false;
		}
	}
</script>

<Dialog.Root bind:open>
	<Dialog.Content class="sm:max-w-lg">
		<Dialog.Header>
			<Dialog.Title>Add custom GGUF</Dialog.Title>
			<Dialog.Description>Pick a GGUF file from a public Hugging Face repo.</Dialog.Description>
		</Dialog.Header>

		<div class="flex flex-col gap-4">
			<div class="flex flex-col gap-2">
				<Label for="hf-repo">HF repo</Label>
				<Input bind:value={hfRepo} id="hf-repo" placeholder="username/repo" />
			</div>

			<div class="flex flex-col gap-2">
				<Label for="hf-file">Model file</Label>
				<Select.Root bind:value={hfFile} disabled={modelFiles.length === 0} type="single">
					<Select.Trigger class="w-full" id="hf-file"
						>{hfFile || 'Select a model file'}</Select.Trigger
					>
					<Select.Content>
						{#each modelFiles as f (f)}
							<Select.Item label={f} value={f}>{f}</Select.Item>
						{/each}
					</Select.Content>
				</Select.Root>
			</div>

			{#if mmprojFiles.length > 0}
				<div class="flex flex-col gap-2">
					<Label for="hf-mmproj">mmproj file (optional, for image or audio input)</Label>
					<Select.Root bind:value={hfMmprojFile} type="single">
						<Select.Trigger class="w-full" id="hf-mmproj">{hfMmprojFile || 'None'}</Select.Trigger>
						<Select.Content>
							<Select.Item label="None" value="">None</Select.Item>
							{#each mmprojFiles as f (f)}
								<Select.Item label={f} value={f}>{f}</Select.Item>
							{/each}
						</Select.Content>
					</Select.Root>
				</div>
			{/if}

			{#if err}
				<p class="text-body-sm text-fg-danger">Error: {err}</p>
			{/if}
		</div>

		<Dialog.Footer>
			<Button disabled={adding} onclick={() => (open = false)} variant="outline">Close</Button>
			<Button disabled={adding || hfRepo.trim().length < 2 || !hfFile} onclick={onSubmit}>
				{#if adding}
					<LoaderCircle class="animate-spin" />
				{/if}
				Add model
			</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
