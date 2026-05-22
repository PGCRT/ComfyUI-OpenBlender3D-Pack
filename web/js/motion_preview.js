/**
 * ComfyUI HY-Motion - Animation Preview Widget
 * Interactive viewer for motion data with GLB export
 */

import { app } from "../../../../scripts/app.js";
import { VIEWER_HTML } from "./viewer_inline.js";

app.registerExtension({
    name: "hymotion.motionpreview",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "HYMotionPreviewAnimation") {

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Create iframe for motion viewer
                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.height = "100%";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#1a1a2e";
                iframe.style.display = "block";

                // Create blob URL from inline HTML
                const blob = new Blob([VIEWER_HTML], { type: 'text/html' });
                const blobUrl = URL.createObjectURL(blob);
                iframe.src = blobUrl;

                iframe.addEventListener('load', () => {
                    iframe._blobUrl = blobUrl;
                });

                iframe.onerror = (e) => {
                    console.error('[HY-Motion] Iframe failed to load:', e);
                };

                // Add widget with min height constraint
                const widget = this.addDOMWidget("preview", "MOTION_PREVIEW", iframe, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                widget.computeSize = function(width) {
                    const w = width || 512;
                    const h = w * 1.2;
                    return [w, h];
                };

                widget.element = iframe;
                this.motionViewerIframe = iframe;
                this.motionViewerReady = false;

                // Listen for ready message
                const onMessage = (event) => {
                    if (event.data && event.data.type === 'VIEWER_READY') {
                        this.motionViewerReady = true;
                    }
                };
                window.addEventListener('message', onMessage.bind(this));

                // Helper function to send resize message to iframe
                const notifyIframeResize = () => {
                    if (iframe.contentWindow) {
                        const rect = iframe.getBoundingClientRect();
                        iframe.contentWindow.postMessage({
                            type: 'RESIZE',
                            width: rect.width,
                            height: rect.height
                        }, '*');
                    }
                };

                // Handle node resize
                this.onResize = function(size) {
                    // Check if we're in vueNodes mode
                    // In vueNodes mode, the DOM widget is rendered inside a Vue component
                    // and we should NOT manually set the iframe height (causes infinite loop)
                    const isVueNodes = iframe.closest('[data-node-id]') !== null ||
                                       document.querySelector('.vue-graph-canvas') !== null;

                    if (!isVueNodes && size && size[1]) {
                        // Only in litegraph mode: manually set iframe height
                        const nodeHeight = size[1];
                        const headerHeight = 70;
                        const availableHeight = Math.max(200, nodeHeight - headerHeight);
                        iframe.style.height = availableHeight + 'px';
                    }

                    // Use requestAnimationFrame to ensure DOM has updated
                    requestAnimationFrame(() => {
                        notifyIframeResize();
                    });
                };

                // Handle resize in vueNodes mode using ResizeObserver with debounce
                let resizeTimeout = null;
                let lastSize = { width: 0, height: 0 };
                const resizeObserver = new ResizeObserver((entries) => {
                    const entry = entries[0];
                    const newWidth = entry.contentRect.width;
                    const newHeight = entry.contentRect.height;

                    // Only trigger if size actually changed significantly (avoid loops)
                    if (Math.abs(newWidth - lastSize.width) < 1 && Math.abs(newHeight - lastSize.height) < 1) {
                        return;
                    }
                    lastSize = { width: newWidth, height: newHeight };

                    // Debounce to prevent rapid updates
                    if (resizeTimeout) {
                        clearTimeout(resizeTimeout);
                    }
                    resizeTimeout = setTimeout(() => {
                        notifyIframeResize();
                    }, 50);
                });
                resizeObserver.observe(iframe);

                // Clean up observer when node is removed
                const originalOnRemoved = this.onRemoved;
                this.onRemoved = function() {
                    resizeObserver.disconnect();
                    if (resizeTimeout) {
                        clearTimeout(resizeTimeout);
                    }
                    if (iframe._blobUrl) {
                        URL.revokeObjectURL(iframe._blobUrl);
                    }
                    if (originalOnRemoved) {
                        originalOnRemoved.apply(this, arguments);
                    }
                };

                this.setSize([512, 614]);

                // Handle execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    onExecuted?.apply(this, arguments);

                    // The message contains motion data
                    if (message?.motion_data && message.motion_data[0]) {
                        const motionDataStr = message.motion_data[0];

                        try {
                            const motionData = JSON.parse(motionDataStr);

                            const sendMessage = () => {
                                if (iframe.contentWindow) {
                                    iframe.contentWindow.postMessage({
                                        type: "LOAD_MOTION",
                                        motionData: motionData,
                                        timestamp: Date.now()
                                    }, "*");
                                } else {
                                    console.error("[HY-Motion] Iframe contentWindow not available");
                                }
                            };

                            if (this.motionViewerReady) {
                                sendMessage();
                            } else {
                                const checkReady = setInterval(() => {
                                    if (this.motionViewerReady) {
                                        clearInterval(checkReady);
                                        sendMessage();
                                    }
                                }, 50);

                                setTimeout(() => {
                                    clearInterval(checkReady);
                                    if (!this.motionViewerReady) {
                                        sendMessage();
                                    }
                                }, 2000);
                            }
                        } catch (e) {
                            console.error("[HY-Motion] Failed to parse motion data:", e);
                        }
                    }
                };

                return r;
            };
        }
    }
});
