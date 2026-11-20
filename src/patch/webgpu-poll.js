// Firefox resolves WebGPU promises (onSubmittedWorkDone, mapAsync) only on a 100 ms timer: https://bugzilla.mozilla.org/show_bug.cgi?id=1870699
// A read back of a WebGPU canvas makes Firefox poll the device at once, and the poll also resolves our pending promises.
// So after each wait starts, we draw a 1x1 WebGPU canvas into a 2D canvas. See "Patches" in README-dev.md
(() => {
  if (
    typeof GPUAdapter === 'undefined' ||
    !/Firefox\//.test(navigator.userAgent)
  ) {
    return;
  }

  let device = null;
  let gpuCanvas = null;
  let gpuCtx = null;
  let ctx2d = null;

  const requestDevice = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function (...args) {
    device = await requestDevice.apply(this, args);
    return device;
  };

  const poll = () => {
    if (!device) {
      return;
    }
    try {
      if (!gpuCanvas) {
        gpuCanvas = new OffscreenCanvas(1, 1);
        gpuCtx = gpuCanvas.getContext('webgpu');
        gpuCtx.configure({
          device,
          format: navigator.gpu.getPreferredCanvasFormat(),
          alphaMode: 'opaque',
        });
        ctx2d = new OffscreenCanvas(1, 1).getContext('2d');
      }
      // without new work on the canvas, Firefox gives the last frame and does not poll
      const encoder = device.createCommandEncoder();
      encoder
        .beginRenderPass({
          colorAttachments: [
            {
              view: gpuCtx.getCurrentTexture().createView(),
              loadOp: 'clear',
              storeOp: 'store',
            },
          ],
        })
        .end();
      device.queue.submit([encoder.finish()]);
      ctx2d.drawImage(gpuCanvas, 0, 0);
    } catch (e) {
      // stop polling, the waits use the 100 ms timer again
      console.warn('wllama: WebGPU poll patch disabled:', e);
      device = null;
    }
  };

  const onSubmittedWorkDone = GPUQueue.prototype.onSubmittedWorkDone;
  GPUQueue.prototype.onSubmittedWorkDone = function (...args) {
    const promise = onSubmittedWorkDone.apply(this, args);
    poll();
    return promise;
  };

  const mapAsync = GPUBuffer.prototype.mapAsync;
  GPUBuffer.prototype.mapAsync = function (...args) {
    const promise = mapAsync.apply(this, args);
    poll();
    return promise;
  };
})();
