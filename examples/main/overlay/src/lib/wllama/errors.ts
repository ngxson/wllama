import { toast } from 'svelte-sonner';

export const errorMessage = (e: unknown) => (e instanceof Error ? e.message : String(e));

// shows an error from wllama as a toast; a stop by the user (abort) is not an error
export function toastError(title: string, e: unknown) {
	if ((e as Error)?.name === 'AbortError') return;
	console.error(`[wllama] ${title}:`, e);
	toast.error(title, { description: errorMessage(e) });
}
