/**
 * SpotCheck — 前端交互逻辑
 *
 * 功能：图片上传（选择/拖拽）、调用识别 API、渲染结果、参数调优控制
 */

// ============================================================
// DOM 元素
// ============================================================
const $ = (sel) => document.querySelector(sel);

const elements = {
    uploadSection:  $("#upload-section"),
    loadingSection: $("#loading-section"),
    resultSection:  $("#result-section"),
    errorSection:   $("#error-section"),
    dropZone:       $("#drop-zone"),
    fileInput:      $("#file-input"),
    loadingStep:    $("#loading-step"),
    errorMessage:   $("#error-message"),
    originalImage:  $("#original-image"),
    annotatedImage: $("#annotated-image"),
    paramsGrid:     $("#params-grid"),
    emptyParams:    $("#empty-params"),
    rawTextsSection:$("#raw-texts-section"),
    rawTextsList:   $("#raw-texts-list"),
};


// ============================================================
// 状态管理
// ============================================================
let isProcessing = false;
let currentFile = null; // 缓存当前图片文件，以便不重新选择文件即可调整参数
let debounceTimer = null;

// 所有参数的默认值
const PARAM_DEFAULTS = {
    "param-grayscale_enabled": false,
    "param-invert_mode": "none",
    "param-invert_black_bg_thresh": 80,
    "param-contrast_mode": "none",
    "param-clahe_clip_limit": 2.5,
    "param-clahe_grid_size": 32,
    "param-linear_alpha": 1.0,
    "param-linear_beta": 0,
    "param-gamma_enabled": false,
    "param-gamma_value": 1.0,
    "param-deblur_enabled": false,
    "param-deblur_direction": "both",
    "param-deblur_strength": 1.0,
    "param-sharpen_enabled": false,
    "param-sharpen_mode": "laplacian",
    "param-sharpen_strength": 1.0,
    "param-denoise_enabled": false,
    "param-denoise_strength": 15,
    "param-morphology_enabled": false,
    "param-morphology_type": "open",
    "param-morphology_size": 2,
    "param-max_size": 960,
    "param-screen_black_threshold": 0.4,
    "param-text_det_thresh": 0.3,
    "param-text_det_box_thresh": 0.5,
    "param-text_det_limit_side_len": 640,
    "param-text_recognition_batch_size": 16,
};

// 浮点数滑块 ID 集合
const FLOAT_SLIDERS = new Set([
    "param-screen_black_threshold",
    "param-text_det_thresh",
    "param-text_det_box_thresh",
]);


// ============================================================
// 初始化
// ============================================================
function init() {
    // 页面加载时获取已保存的参数
    loadSavedParams();

    // 文件选择事件
    elements.fileInput.addEventListener("change", handleFileSelect);

    // 拖拽事件
    const dz = elements.dropZone;
    dz.addEventListener("dragover", (e) => {
        e.preventDefault();
        dz.classList.add("dragover");
    });
    dz.addEventListener("dragleave", () => {
        dz.classList.remove("dragover");
    });
    dz.addEventListener("drop", (e) => {
        e.preventDefault();
        dz.classList.remove("dragover");
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            processFile(files[0]);
        }
    });

    // 点击上传区域（但不是按钮时）触发文件选择
    dz.addEventListener("click", (e) => {
        // 如果点击的是按钮或 label 内部，不重复触发
        if (e.target.closest("label") || e.target.closest("button")) return;
        elements.fileInput.click();
    });

    // 图片点击放大
    elements.originalImage.addEventListener("click", () => showLightbox(elements.originalImage.src));
    elements.annotatedImage.addEventListener("click", () => showLightbox(elements.annotatedImage.src));
}


// ============================================================
// 文件处理
// ============================================================
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        processFile(file);
    }
    // 重置 input 以允许重复选择同一文件
    e.target.value = "";
}

async function processFile(file) {
    if (isProcessing) return;

    // 验证文件类型
    if (!file.type.startsWith("image/")) {
        showError("请选择图片文件（JPG/PNG）");
        return;
    }

    // 验证文件大小（限制 20MB）
    if (file.size > 20 * 1024 * 1024) {
        showError("图片文件过大，请不要超过 20MB");
        return;
    }

    isProcessing = true;
    currentFile = file; // 缓存文件引用
    showSection("loading");
    updateLoadingStep("正在上传图片...");

    try {
        const formData = new FormData();
        formData.append("image", file);

        // 获取当前预处理面板参数
        const params = getPreprocessParams();
        formData.append("params", JSON.stringify(params));

        updateLoadingStep("正在识别中...");

        const response = await fetch("/api/recognize", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `服务器错误 (${response.status})`);
        }

        const result = await response.json();

        if (result.success) {
            renderResults(result.data);
            showSection("result");
            // 初始化字段的显示隐藏状态
            toggleInvertThresh();
            toggleContrastFields();
            toggleGammaFields();
            toggleDeblurFields();
            toggleSharpenFields();
            toggleDenoiseFields();
            toggleMorphologyFields();
        } else {
            throw new Error(result.error || "识别失败");
        }
    } catch (err) {
        console.error("识别失败:", err);
        showError(err.message || "网络错误，请检查服务是否启动");
    } finally {
        isProcessing = false;
    }
}


// ============================================================
// 渲染识别结果
// ============================================================
function renderResults(data) {
    const { parameters, raw_texts, original_image, annotated_image, stats } = data;

    // 统计信息
    $("#stat-total-time").textContent = formatTime(stats.total_time_ms);
    $("#stat-param-count").textContent = stats.param_count;
    $("#stat-text-count").textContent = stats.text_count;
    $("#stat-ocr-time").textContent = formatTime(stats.ocr_time_ms);

    // 图片
    elements.originalImage.src = `data:image/jpeg;base64,${original_image}`;
    elements.annotatedImage.src = `data:image/jpeg;base64,${annotated_image}`;

    // 参数卡片
    renderParameters(parameters);

    // 原始文本
    renderRawTexts(raw_texts);
}

function renderParameters(params) {
    const grid = elements.paramsGrid;
    grid.innerHTML = "";

    if (!params || params.length === 0) {
        elements.emptyParams.classList.remove("hidden");
        return;
    }

    elements.emptyParams.classList.add("hidden");

    params.forEach((param, index) => {
        const card = document.createElement("div");
        card.className = `param-card ${param.source === "standalone" ? "standalone" : ""}`;
        card.style.animationDelay = `${index * 0.06}s`;

        const conf = param.confidence || 0;
        const confPercent = Math.round(conf * 100);
        const confLevel = confPercent >= 90 ? "high" : confPercent >= 70 ? "medium" : "low";

        const sourceLabelMap = {
            "inline": "行内",
            "spatial": "配对",
            "standalone": "独立",
        };

        card.innerHTML = `
            <div class="param-name">
                ${param.name || '<span style="color:var(--text-muted);font-style:italic">未知参数</span>'}
                <span class="source-tag">${sourceLabelMap[param.source] || param.source}</span>
            </div>
            <div class="param-value-row">
                <span class="param-value">${escapeHtml(param.value)}</span>
                ${param.unit ? `<span class="param-unit">${escapeHtml(param.unit)}</span>` : ""}
            </div>
            <div class="param-confidence">
                <div class="confidence-bar">
                    <div class="confidence-fill ${confLevel}" style="width: ${confPercent}%"></div>
                </div>
                <span class="confidence-text">${confPercent}%</span>
            </div>
        `;

        grid.appendChild(card);
    });
}

function renderRawTexts(texts) {
    const list = elements.rawTextsList;
    list.innerHTML = "";

    if (!texts || texts.length === 0) return;

    texts.forEach((item) => {
        const div = document.createElement("div");
        div.className = "raw-text-item";

        const confPercent = Math.round((item.confidence || 0) * 100);

        div.innerHTML = `
            <span class="raw-text-content">${escapeHtml(item.text)}</span>
            <span class="raw-text-conf">${confPercent}%</span>
        `;

        list.appendChild(div);
    });
}


// ============================================================
// UI 状态切换
// ============================================================
function showSection(name) {
    elements.uploadSection.classList.add("hidden");
    elements.loadingSection.classList.add("hidden");
    elements.resultSection.classList.add("hidden");
    elements.errorSection.classList.add("hidden");

    switch (name) {
        case "upload":  elements.uploadSection.classList.remove("hidden"); break;
        case "loading": elements.loadingSection.classList.remove("hidden"); break;
        case "result":  elements.resultSection.classList.remove("hidden"); break;
        case "error":   elements.errorSection.classList.remove("hidden"); break;
    }
}

function updateLoadingStep(text) {
    elements.loadingStep.textContent = text;
}

function showError(message) {
    elements.errorMessage.textContent = message;
    showSection("error");
}

function resetUpload() {
    showSection("upload");
    currentFile = null;
    // 清理之前的结果
    elements.paramsGrid.innerHTML = "";
    elements.rawTextsList.innerHTML = "";
    elements.rawTextsSection.classList.add("hidden");
    
    // 重置表单
    const form = document.getElementById("preprocess-form");
    if (form) {
        form.reset();
        // 延迟触发以使 reset 生效后获取到初始默认值
        setTimeout(() => {
            toggleInvertThresh();
            toggleContrastFields();
            toggleGammaFields();
            toggleDeblurFields();
            toggleSharpenFields();
            toggleDenoiseFields();
            toggleMorphologyFields();
            // 重置滑块显示的数值
            document.querySelectorAll("input[type=range]").forEach(el => {
                const valSpan = document.getElementById("val-" + el.id.replace("param-", ""));
                if (valSpan) {
                    if (FLOAT_SLIDERS.has(el.id)) {
                        valSpan.textContent = parseFloat(el.value).toFixed(2);
                    } else {
                        valSpan.textContent = el.value;
                    }
                }
            });
        }, 50);
    }
}

function toggleRawTexts() {
    elements.rawTextsSection.classList.toggle("hidden");
}


// ============================================================
// 图片灯箱
// ============================================================
function showLightbox(src) {
    const lb = document.createElement("div");
    lb.className = "image-lightbox";
    lb.innerHTML = `<img src="${src}" alt="放大预览">`;
    lb.addEventListener("click", () => lb.remove());
    document.body.appendChild(lb);
}


// ============================================================
// 图像预处理参数获取与动作绑定
// ============================================================
function getPreprocessParams() {
    return {
        // 图像基础参数
        max_size: $("#param-max_size") ? parseInt($("#param-max_size").value, 10) : 960,
        screen_black_threshold: $("#param-screen_black_threshold") ? parseFloat($("#param-screen_black_threshold").value) : 0.4,
        // 预处理参数
        grayscale_enabled: $("#param-grayscale_enabled") ? $("#param-grayscale_enabled").checked : false,
        invert_mode: $("#param-invert_mode") ? $("#param-invert_mode").value : "none",
        invert_black_bg_thresh: $("#param-invert_black_bg_thresh") ? parseFloat($("#param-invert_black_bg_thresh").value) : 80,
        contrast_mode: $("#param-contrast_mode") ? $("#param-contrast_mode").value : "none",
        clahe_clip_limit: $("#param-clahe_clip_limit") ? parseFloat($("#param-clahe_clip_limit").value) : 2.5,
        clahe_grid_size: $("#param-clahe_grid_size") ? parseInt($("#param-clahe_grid_size").value, 10) : 32,
        linear_alpha: $("#param-linear_alpha") ? parseFloat($("#param-linear_alpha").value) : 1.0,
        linear_beta: $("#param-linear_beta") ? parseFloat($("#param-linear_beta").value) : 0,
        gamma_enabled: $("#param-gamma_enabled") ? $("#param-gamma_enabled").checked : false,
        gamma_value: $("#param-gamma_value") ? parseFloat($("#param-gamma_value").value) : 1.0,
        deblur_enabled: $("#param-deblur_enabled") ? $("#param-deblur_enabled").checked : false,
        deblur_direction: $("#param-deblur_direction") ? $("#param-deblur_direction").value : "both",
        deblur_strength: $("#param-deblur_strength") ? parseFloat($("#param-deblur_strength").value) : 1.0,
        sharpen_enabled: $("#param-sharpen_enabled") ? $("#param-sharpen_enabled").checked : false,
        sharpen_mode: $("#param-sharpen_mode") ? $("#param-sharpen_mode").value : "laplacian",
        sharpen_strength: $("#param-sharpen_strength") ? parseFloat($("#param-sharpen_strength").value) : 1.0,
        denoise_enabled: $("#param-denoise_enabled") ? $("#param-denoise_enabled").checked : false,
        denoise_strength: $("#param-denoise_strength") ? parseInt($("#param-denoise_strength").value, 10) : 15,
        morphology_enabled: $("#param-morphology_enabled") ? $("#param-morphology_enabled").checked : false,
        morphology_type: $("#param-morphology_type") ? $("#param-morphology_type").value : "open",
        morphology_size: $("#param-morphology_size") ? parseInt($("#param-morphology_size").value, 10) : 2,
        // OCR 引擎参数
        text_det_thresh: $("#param-text_det_thresh") ? parseFloat($("#param-text_det_thresh").value) : 0.3,
        text_det_box_thresh: $("#param-text_det_box_thresh") ? parseFloat($("#param-text_det_box_thresh").value) : 0.5,
        text_det_limit_side_len: $("#param-text_det_limit_side_len") ? parseInt($("#param-text_det_limit_side_len").value, 10) : 640,
        text_recognition_batch_size: $("#param-text_recognition_batch_size") ? parseInt($("#param-text_recognition_batch_size").value, 10) : 16,
    };
}

window.updateSliderVal = function(el) {
    const valSpan = document.getElementById("val-" + el.id.replace("param-", ""));
    if (valSpan) {
        valSpan.textContent = el.value;
    }
};

window.updateMaxSizeDisplay = function(el) {
    const valSpan = document.getElementById("val-max_size");
    if (valSpan) {
        valSpan.textContent = parseInt(el.value, 10) === 0 ? "不限制 (原图)" : el.value;
    }
};

window.updateSliderValFloat = function(el) {
    const valSpan = document.getElementById("val-" + el.id.replace("param-", ""));
    if (valSpan) {
        valSpan.textContent = parseFloat(el.value).toFixed(2);
    }
};

window.toggleInvertThresh = function() {
    const mode = $("#param-invert_mode") ? $("#param-invert_mode").value : "none";
    const threshRow = $("#row-invert_thresh");
    if (threshRow) {
        if (mode === "local_black_bg") {
            threshRow.classList.remove("hidden");
        } else {
            threshRow.classList.add("hidden");
        }
    }
};

window.toggleContrastFields = function() {
    const mode = $("#param-contrast_mode") ? $("#param-contrast_mode").value : "none";
    const claheClipRow = $("#row-clahe_clip");
    const linearRow = $("#row-linear_fields");
    
    if (claheClipRow) claheClipRow.classList.add("hidden");
    if (linearRow) linearRow.classList.add("hidden");
    
    if (mode === "clahe" && claheClipRow) {
        claheClipRow.classList.remove("hidden");
    } else if (mode === "linear" && linearRow) {
        linearRow.classList.remove("hidden");
    }
};

window.toggleGammaFields = function() {
    const enabled = $("#param-gamma_enabled") ? $("#param-gamma_enabled").checked : false;
    const row = $("#row-gamma_val");
    if (row) {
        if (enabled) {
            row.classList.remove("hidden");
        } else {
            row.classList.add("hidden");
        }
    }
};

window.toggleDeblurFields = function() {
    const enabled = $("#param-deblur_enabled") ? $("#param-deblur_enabled").checked : false;
    const row = $("#row-deblur_fields");
    if (row) {
        if (enabled) {
            row.classList.remove("hidden");
        } else {
            row.classList.add("hidden");
        }
    }
};

window.toggleSharpenFields = function() {
    const enabled = $("#param-sharpen_enabled") ? $("#param-sharpen_enabled").checked : false;
    const row = $("#row-sharpen_fields");
    if (row) {
        if (enabled) {
            row.classList.remove("hidden");
        } else {
            row.classList.add("hidden");
        }
    }
};

window.toggleDenoiseFields = function() {
    const enabled = $("#param-denoise_enabled") ? $("#param-denoise_enabled").checked : false;
    const row = $("#row-denoise_strength");
    if (row) {
        if (enabled) {
            row.classList.remove("hidden");
        } else {
            row.classList.add("hidden");
        }
    }
};

window.toggleMorphologyFields = function() {
    const enabled = $("#param-morphology_enabled") ? $("#param-morphology_enabled").checked : false;
    const row = $("#row-morphology_fields");
    if (row) {
        if (enabled) {
            row.classList.remove("hidden");
        } else {
            row.classList.add("hidden");
        }
    }
};

window.autoSubmitIfEnabled = function() {
    const autoSubmitCheckbox = $("#auto-submit-checkbox");
    const autoSubmit = autoSubmitCheckbox ? autoSubmitCheckbox.checked : false;
    if (autoSubmit && currentFile) {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            window.reRunPreprocessing();
        }, 400); // 400ms 防抖，避免拖拽滑动条时疯狂请求后台
    }
};

window.reRunPreprocessing = async function() {
    if (!currentFile || isProcessing) return;
    
    isProcessing = true;
    const reAnalyzeBtn = document.getElementById("re-analyze-btn");
    let originalText = "";
    if (reAnalyzeBtn) {
        originalText = reAnalyzeBtn.innerHTML;
        reAnalyzeBtn.disabled = true;
        reAnalyzeBtn.innerHTML = `<span class="spinner-ring" style="width:14px;height:14px;position:relative;display:inline-block;margin-right:8px;border-top-color:#fff;vertical-align:middle;"></span> 正在重新解析...`;
    }
    
    try {
        const formData = new FormData();
        formData.append("image", currentFile);
        
        const params = getPreprocessParams();
        formData.append("params", JSON.stringify(params));
        
        const response = await fetch("/api/recognize", {
            method: "POST",
            body: formData,
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `服务器错误 (${response.status})`);
        }
        
        const result = await response.json();
        if (result.success) {
            renderResults(result.data);
        } else {
            throw new Error(result.error || "识别失败");
        }
    } catch (err) {
        console.error("重新识别失败:", err);
        alert("重新识别失败: " + err.message);
    } finally {
        isProcessing = false;
        if (reAnalyzeBtn) {
            reAnalyzeBtn.disabled = false;
            reAnalyzeBtn.innerHTML = originalText;
        }
    }
};

// 所有参数的默认值（已在文件顶部定义）

window.resetParamsToDefault = function() {
    // 重置复选框
    document.querySelectorAll('#preprocess-form input[type="checkbox"]').forEach(el => {
        const defVal = PARAM_DEFAULTS[el.id];
        el.checked = !!defVal;
    });

    // 重置下拉菜单
    document.querySelectorAll('#preprocess-form select').forEach(el => {
        const defVal = PARAM_DEFAULTS[el.id];
        if (defVal !== undefined) el.value = defVal;
    });

    // 重置滑块
    document.querySelectorAll('#preprocess-form input[type="range"]').forEach(el => {
        const defVal = PARAM_DEFAULTS[el.id];
        if (defVal !== undefined) {
            el.value = defVal;
            const valSpan = document.getElementById("val-" + el.id.replace("param-", ""));
            if (valSpan) {
                if (FLOAT_SLIDERS.has(el.id)) {
                    valSpan.textContent = parseFloat(defVal).toFixed(2);
                } else if (el.id === "param-max_size" && parseInt(defVal, 10) === 0) {
                    valSpan.textContent = "不限制 (原图)";
                } else {
                    valSpan.textContent = defVal;
                }
            }
        }
    });

    // 更新字段显示状态
    toggleInvertThresh();
    toggleContrastFields();
    toggleGammaFields();
    toggleDeblurFields();
    toggleSharpenFields();
    toggleDenoiseFields();
    toggleMorphologyFields();

    // 更新保存状态提示
    updateSavedStatus(false, "未保存自定义参数");

    // 触发自动重新解析
    autoSubmitIfEnabled();
};

// ============================================================
// 智能自动调优
// ============================================================
window.autoTuneParams = async function() {
    if (!currentFile) {
        alert("请先上传一张图片，然后点击智能调优");
        return;
    }
    if (isProcessing) return;

    const btn = document.getElementById("auto-tune-btn");
    const origText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner-ring" style="width:14px;height:14px;position:relative;display:inline-block;margin-right:8px;border-top-color:#333;vertical-align:middle;"></span> 正在分析图片...`;

    try {
        const formData = new FormData();
        formData.append("image", currentFile);

        const response = await fetch("/api/auto-tune", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) throw new Error(`服务器错误 (${response.status})`);
        const result = await response.json();

        if (result.success) {
            applyParamsToForm(result.data);
            // 自动触发重新解析
            autoSubmitIfEnabled();
        } else {
            throw new Error(result.message || "自动调优失败");
        }
    } catch (err) {
        console.error("自动调优失败:", err);
        alert("自动调优失败: " + err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = origText;
    }
};

// 将参数应用到表单控件
function applyParamsToForm(params) {
    if (!params) return;

    // 复选框
    const checkboxKeys = ["grayscale_enabled", "gamma_enabled", "deblur_enabled",
                          "sharpen_enabled", "denoise_enabled", "morphology_enabled"];
    checkboxKeys.forEach(key => {
        const el = document.getElementById("param-" + key);
        if (el && key in params) el.checked = !!params[key];
    });

    // 下拉菜单
    const selectKeys = ["invert_mode", "contrast_mode", "deblur_direction",
                        "sharpen_mode", "morphology_type"];
    selectKeys.forEach(key => {
        const el = document.getElementById("param-" + key);
        if (el && key in params) el.value = params[key];
    });

    // 滑块
    const sliderKeys = ["invert_black_bg_thresh", "clahe_clip_limit", "clahe_grid_size",
                         "linear_alpha", "linear_beta", "gamma_value", "deblur_strength",
                         "sharpen_strength", "denoise_strength", "morphology_size",
                         "max_size", "screen_black_threshold", "text_det_thresh",
                         "text_det_box_thresh", "text_det_limit_side_len", "text_recognition_batch_size"];
    sliderKeys.forEach(key => {
        const el = document.getElementById("param-" + key);
        if (el && key in params) {
            el.value = params[key];
            const valSpan = document.getElementById("val-" + key);
            if (valSpan) {
                if (FLOAT_SLIDERS.has("param-" + key)) {
                    valSpan.textContent = parseFloat(params[key]).toFixed(2);
                } else if (key === "max_size" && parseInt(params[key], 10) === 0) {
                    valSpan.textContent = "不限制 (原图)";
                } else {
                    valSpan.textContent = params[key];
                }
            }
        }
    });

    // 更新字段显示/隐藏状态
    toggleInvertThresh();
    toggleContrastFields();
    toggleGammaFields();
    toggleDeblurFields();
    toggleSharpenFields();
    toggleDenoiseFields();
    toggleMorphologyFields();
}

// ============================================================
// 永久保存参数
// ============================================================
window.saveParamsPermanent = async function() {
    const btn = document.getElementById("save-params-btn");
    const origText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner-ring" style="width:14px;height:14px;position:relative;display:inline-block;margin-right:8px;border-top-color:#fff;vertical-align:middle;"></span> 保存中...`;

    try {
        const params = getPreprocessParams();
        const formData = new FormData();
        formData.append("params", JSON.stringify(params));

        const response = await fetch("/api/save-params", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) throw new Error(`服务器错误 (${response.status})`);
        const result = await response.json();

        if (result.success) {
            updateSavedStatus(true, "已保存为永久默认参数");
        } else {
            throw new Error(result.message || "保存失败");
        }
    } catch (err) {
        console.error("保存失败:", err);
        alert("保存失败: " + err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = origText;
    }
};

// 更新保存状态提示
function updateSavedStatus(saved, text) {
    const dot = document.getElementById("saved-dot");
    const statusText = document.getElementById("saved-status-text");
    if (dot) {
        dot.className = "saved-dot " + (saved ? "saved" : "unsaved");
    }
    if (statusText) {
        statusText.textContent = text;
    }
}

// 页面加载时获取已保存的参数
async function loadSavedParams() {
    try {
        const response = await fetch("/api/saved-params");
        if (!response.ok) return;
        const result = await response.json();
        if (result.success && result.data) {
            applyParamsToForm(result.data);
            updateSavedStatus(true, "已加载保存的默认参数");
        }
    } catch (err) {
        console.warn("加载已保存参数失败:", err);
    }
}


// ============================================================
// 工具函数
// ============================================================
function formatTime(ms) {
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}


// ============================================================
// 启动
// ============================================================
document.addEventListener("DOMContentLoaded", init);
