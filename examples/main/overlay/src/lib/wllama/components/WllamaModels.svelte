<script lang="ts">
	import * as actions from '../actions';
	import { DEFAULT_LOAD_PARAMS, type LoadParams } from '../config';
	import { getWllamaServer } from '../wllama-server';
	import { wllamaStore } from '../store.svelte';
	import WllamaAddModelDialog from './WllamaAddModelDialog.svelte';
	import WllamaModelCard from './WllamaModelCard.svelte';
	import WllamaRuntimeInfo from './WllamaRuntimeInfo.svelte';
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import * as Collapsible from '@ggml-org/llama-ui/ui/collapsible';
	import { Input } from '@ggml-org/llama-ui/ui/input';
	import { Label } from '@ggml-org/llama-ui/ui/label';
	import { ChevronDown, ChevronRight, Plus } from '@lucide/svelte';
	import { DialogConfirmation } from '#lib/components/app/dialogs/index.ts';
	import { modelsStore } from '#lib/stores/index.ts';
	import { onMount } from 'svelte';

	onMount(() => {
		wllamaStore.start();
		// same as the upstream models page: the app store ignores the status of the models it has not listed
		if (modelsStore.models.length > 0) {
			void modelsStore.fetchRouterModels();
		} else {
			modelsStore.fetch().catch((error) => console.error('Unable to load models:', error));
		}
	});

	let showAddDialog = $state(false);
	let paramsOpen = $state(false);
	let pendingConfirm = $state<{ title: string; description: string; onConfirm: () => void } | null>(
		null
	);

	const confirm = (title: string, description: string, onConfirm: () => void) => {
		pendingConfirm = { description, onConfirm, title };
	};

	// same rule as before: no download, load or removal while a model is loaded or busy
	const blocked = $derived(!!wllamaStore.loadedId || wllamaStore.busy || wllamaStore.isDownloading);
	const paramsLocked = $derived(!!wllamaStore.loadedId || wllamaStore.busy);
	const customModels = $derived(wllamaStore.models.filter((m) => m.isUserAdded));
	const presetModels = $derived(wllamaStore.models.filter((m) => !m.isUserAdded));

	const PARAM_FIELDS: {
		key: keyof LoadParams;
		label: string;
		min: number;
		placeholder?: string;
	}[] = [
		{ key: 'nThreads', label: 'Threads', min: 1, placeholder: '(auto)' },
		{ key: 'nContext', label: 'Context size', min: 128 },
		{ key: 'nBatch', label: 'Batch size', min: 1 }
	];

	function setParam(key: keyof LoadParams, value: string) {
		const n = parseInt(value, 10);
		getWllamaServer().setLoadParams({
			...wllamaStore.loadParams,
			[key]: Number.isFinite(n) ? n : -1
		});
	}
</script>

<div class="flex flex-col gap-8 pb-10">
	<WllamaRuntimeInfo />

	<Collapsible.Root bind:open={paramsOpen}>
		<Collapsible.Trigger class="flex items-center gap-2">
			{#if paramsOpen}
				<ChevronDown class="size-icon-md" />
			{:else}
				<ChevronRight class="size-icon-md" />
			{/if}
			<h2 class="text-title">Load parameters</h2>
		</Collapsible.Trigger>

		<Collapsible.Content class="flex flex-col gap-3 pt-3">
			<p class="text-body-sm text-muted-foreground">
				Used the next time a model is loaded. The sampling settings (temperature, ...) are in the
				chat settings.
			</p>

			<div class="grid gap-4 sm:grid-cols-3">
				{#each PARAM_FIELDS as field (field.key)}
					{@const value = wllamaStore.loadParams[field.key]}
					<div class="flex flex-col gap-2">
						<Label for="param-{field.key}">{field.label}</Label>
						<Input
							disabled={paramsLocked}
							id="param-{field.key}"
							min={field.min}
							onchange={(e) => setParam(field.key, e.currentTarget.value)}
							placeholder={field.placeholder}
							type="number"
							value={value < 1 ? '' : value}
						/>
					</div>
				{/each}
			</div>

			<div class="flex flex-wrap gap-2">
				<Button
					disabled={paramsLocked}
					onclick={() => getWllamaServer().setLoadParams(DEFAULT_LOAD_PARAMS)}
					size="sm"
					variant="outline"
				>
					Reset params
				</Button>
				<Button
					disabled={blocked}
					onclick={() =>
						confirm(
							'Clear the cache?',
							'This removes all the downloaded model files.',
							actions.clearCache
						)}
					size="sm"
					variant="outline"
				>
					Clear cache
				</Button>
			</div>
		</Collapsible.Content>
	</Collapsible.Root>

	<section class="flex flex-col gap-3">
		<div class="flex items-center gap-4">
			<h2 class="text-title">Custom models</h2>
			<Button onclick={() => (showAddDialog = true)} size="sm" variant="tertiary">
				<Plus />
				Add GGUF
			</Button>
		</div>

		{#each customModels as model (model.id)}
			<WllamaModelCard {blocked} {confirm} {model} />
		{:else}
			<p class="text-body-sm text-muted-foreground">No custom model yet.</p>
		{/each}
	</section>

	<section class="flex flex-col gap-3">
		<h2 class="text-title">Recommended models</h2>

		{#each presetModels as model (model.id)}
			<WllamaModelCard {blocked} {confirm} {model} />
		{/each}
	</section>
</div>

<WllamaAddModelDialog bind:open={showAddDialog} />

<DialogConfirmation
	description={pendingConfirm?.description ?? ''}
	onCancel={() => (pendingConfirm = null)}
	onConfirm={() => {
		pendingConfirm?.onConfirm();
		pendingConfirm = null;
	}}
	open={!!pendingConfirm}
	title={pendingConfirm?.title ?? ''}
	variant="destructive"
/>
