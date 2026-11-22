<script lang="ts">
	import * as actions from '../actions';
	import type { ModelEntry } from '../wllama-server';
	import { wllamaStore } from '../store.svelte';
	import {
		CAPABILITY_ICONS,
		CAPABILITY_LABELS,
		MODALITY_ICONS,
		MODALITY_LABELS,
		ModelCapability,
		ModelModality
	} from '@ggml-org/llama-ui/models';
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import {
		Check,
		Download,
		Eject,
		LoaderCircle,
		MessageSquare,
		Play,
		Trash,
		X
	} from '@lucide/svelte';
	import { formatFileSize } from '#lib/utils/index.ts';

	interface Props {
		model: ModelEntry;
		// disables the actions that start a download, a load or a removal
		blocked: boolean;
		confirm: (title: string, description: string, onConfirm: () => void) => void;
	}

	let { blocked, confirm, model }: Props = $props();

	const repo = $derived(
		model.url
			.replace(/^https?:\/\/(huggingface\.co|hf\.co)\/+/, '')
			.split('/')
			.slice(0, 2)
			.join('/')
	);
	const fileName = $derived(
		(model.url.split('/').pop() ?? model.url).replace(/-\d{5}-of-\d{5}/, '-(shards)')
	);
	const percent = $derived(Math.round(model.downloadProgress * 100));
	const isDownloading = $derived(model.downloadProgress >= 0);
	const info = $derived(model.status === 'loaded' ? wllamaStore.runtimeInfo : null);
	// the icons and labels of the upstream models manager, one color each; reasoning is not shown
	const capabilities = $derived([
		...(model.tools
			? [
					{
						color: 'text-green-600 dark:text-green-400',
						icon: CAPABILITY_ICONS[ModelCapability.TOOL_USE],
						label: CAPABILITY_LABELS[ModelCapability.TOOL_USE]
					}
				]
			: []),
		...(model.modalities.includes('image')
			? [
					{
						color: 'text-sky-600 dark:text-sky-400',
						icon: MODALITY_ICONS[ModelModality.VISION],
						label: MODALITY_LABELS[ModelModality.VISION]
					}
				]
			: []),
		...(model.modalities.includes('audio')
			? [
					{
						color: 'text-amber-600 dark:text-amber-400',
						icon: MODALITY_ICONS[ModelModality.AUDIO],
						label: MODALITY_LABELS[ModelModality.AUDIO]
					}
				]
			: [])
	]);
</script>

<div
	class="card-surface flex flex-col gap-3 rounded-xl p-4 sm:flex-row sm:items-center {model.status ===
	'loaded'
		? 'ring-2 ring-(--action-primary)'
		: ''}"
>
	<div class="flex min-w-0 grow flex-col gap-1">
		<!-- the capability icons follow the name, like in the upstream models manager -->
		<div class="flex min-w-0 items-center gap-1.25">
			<span class="truncate font-medium" title={model.url}>{fileName}</span>

			{#each capabilities as capability (capability.label)}
				<Tooltip.Root>
					<Tooltip.Trigger>
						<capability.icon class="size-icon-sm {capability.color}" />
					</Tooltip.Trigger>
					<Tooltip.Content>
						<p>{capability.label}</p>
					</Tooltip.Content>
				</Tooltip.Root>
			{/each}
		</div>

		<div class="flex flex-wrap items-center gap-x-2 gap-y-1 text-body-sm text-muted-foreground">
			<span>{repo}</span>
			<span>&middot;</span>
			<span>{formatFileSize(model.size)}</span>

			{#if isDownloading}
				<span>&middot; Downloaded: {percent}%</span>
			{/if}
		</div>

		{#if info}
			<div class="flex flex-wrap gap-4 text-body-sm">
				{#each [['Multithread', info.isMultithread], ['Chat template', info.hasChatTemplate]] as const as [label, on] (label)}
					<span class="flex items-center gap-1">
						{#if on}
							<Check class="size-icon-sm text-green-500" />
						{:else}
							<X class="size-icon-sm text-red-500" />
						{/if}
						{label}
					</span>
				{/each}
				<span class="text-muted-foreground">{info.nThreads} threads, context {info.nCtx}</span>
			</div>
		{/if}

		{#if isDownloading}
			<div class="h-1.5 w-full overflow-hidden rounded-full bg-(--line-default)">
				<div class="h-full bg-(--action-primary) transition-all" style="width: {percent}%"></div>
			</div>
		{:else if model.status === 'loading'}
			<div class="h-1.5 w-full overflow-hidden rounded-full bg-(--line-default)">
				<div class="h-full w-1/3 animate-pulse bg-(--action-primary)"></div>
			</div>
		{/if}
	</div>

	<div class="flex shrink-0 items-center gap-2">
		{#if isDownloading}
			<LoaderCircle class="size-icon-md animate-spin" />
			<Button onclick={() => actions.cancelDownload(model.id)} size="sm" variant="outline"
				>Cancel</Button
			>
		{:else if !model.cached}
			<Button disabled={blocked} onclick={() => actions.downloadModel(model.id)} size="sm">
				<Download />
				Download
			</Button>
			{#if model.isUserAdded}
				<Button
					disabled={blocked}
					onclick={() =>
						confirm('Remove this model from the list?', model.url, () =>
							actions.removeCustomModel(model.id)
						)}
					size="icon"
					title="Remove from the list"
					variant="outline"
				>
					<X />
				</Button>
			{/if}
		{:else if model.status === 'loaded'}
			<Button onclick={() => actions.startChat(model.id)} size="sm">
				<MessageSquare />
				Start chat
			</Button>
			<Button onclick={() => actions.unloadModel(model.id)} size="sm" variant="outline">
				<Eject />
				Unload
			</Button>
		{:else if model.status === 'loading'}
			<LoaderCircle class="size-icon-md animate-spin" />
		{:else}
			<Button disabled={blocked} onclick={() => actions.loadModel(model.id)} size="sm">
				<Play />
				Load model
			</Button>
			<Button
				disabled={blocked}
				onclick={() =>
					confirm('Remove this model from the cache?', model.url, () =>
						actions.removeFromCache(model.id)
					)}
				size="icon"
				title="Remove from the cache"
				variant="outline"
			>
				<Trash />
			</Button>
		{/if}
	</div>
</div>
