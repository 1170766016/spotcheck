/**
 * SpotCheck AI — 前端交互逻辑
 *
 * 功能：图片上传（选择/拖拽）、调用识别 API、渲染结果
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


// ============================================================
// 初始化
// ============================================================
function init() {
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
    showSection("loading");
    updateLoadingStep("正在上传图片...");

    try {
        const formData = new FormData();
        formData.append("image", file);

        updateLoadingStep("AI 正在识别中...");

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
    // 清理之前的结果
    elements.paramsGrid.innerHTML = "";
    elements.rawTextsList.innerHTML = "";
    elements.rawTextsSection.classList.add("hidden");
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
