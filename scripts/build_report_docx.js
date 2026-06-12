const fs = require("fs");
const path = require("path");
require("module").Module._initPaths();

const {
  AlignmentType,
  BorderStyle,
  Document,
  ImageRun,
  Packer,
  Paragraph,
  Table,
  TableCell,
  TableLayoutType,
  TableRow,
  TextRun,
  VerticalAlign,
  WidthType,
} = require("docx");

const ROOT = "C:/Users/ben04/Desktop/speaker_id_project";
const OUT = path.join(ROOT, "Speaker_ID_Project_Report.docx");

function readCsv(file) {
  const text = fs.readFileSync(file, "utf8").trim();
  const lines = text.split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = [];
    let cur = "";
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') inQuotes = !inQuotes;
      else if (ch === "," && !inQuotes) {
        values.push(cur);
        cur = "";
      } else cur += ch;
    }
    values.push(cur);
    return Object.fromEntries(headers.map((h, i) => [h, values[i] ?? ""]));
  });
}

const pct = (x) => `${(parseFloat(x) * 100).toFixed(2)}%`;
const num = (x) => parseFloat(x).toFixed(4);
const modelName = (m) => m.replace("baseline", "Baseline").replace("gmm_", "GMM-").replace("ecapa", "ECAPA");

const closed = readCsv(path.join(ROOT, "results/metrics/closed_set_metrics.csv"));
const open = readCsv(path.join(ROOT, "results/metrics/open_set_metrics.csv"));
const comp = readCsv(path.join(ROOT, "results/metrics/embedding_compression_metrics.csv"));

const normalRun = (text) => new TextRun({ text, font: "Times New Roman", size: 18 });
const boldRun = (text, size = 18) => new TextRun({ text, font: "Times New Roman", size, bold: true });
const italicRun = (text, size = 16) => new TextRun({ text, font: "Times New Roman", size, italics: true });

function para(text, opts = {}) {
  return new Paragraph({
    children: [opts.bold ? boldRun(text, opts.size ?? 18) : new TextRun({ text, font: "Times New Roman", size: opts.size ?? 18, bold: !!opts.bold, italics: !!opts.italics })],
    alignment: opts.alignment,
    spacing: { before: opts.before ?? 0, after: opts.after ?? 80, line: 220 },
  });
}

function heading(text) {
  return new Paragraph({
    children: [boldRun(text.toUpperCase(), 20)],
    alignment: AlignmentType.CENTER,
    spacing: { before: 180, after: 120 },
  });
}

function caption(text) {
  return new Paragraph({
    children: [italicRun(text, 16)],
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 120 },
  });
}

function table(rows) {
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    layout: TableLayoutType.AUTOFIT,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 1, color: "B7C3D0" },
      bottom: { style: BorderStyle.SINGLE, size: 1, color: "B7C3D0" },
      left: { style: BorderStyle.SINGLE, size: 1, color: "B7C3D0" },
      right: { style: BorderStyle.SINGLE, size: 1, color: "B7C3D0" },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 1, color: "D6DEE8" },
      insideVertical: { style: BorderStyle.SINGLE, size: 1, color: "D6DEE8" },
    },
    rows: rows.map((row, r) =>
      new TableRow({
        tableHeader: r === 0,
        children: row.map(
          (cell) =>
            new TableCell({
              verticalAlign: VerticalAlign.CENTER,
              shading: { fill: r === 0 ? "D9EAF7" : r % 2 === 0 ? "F4F7FB" : "FFFFFF" },
              margins: { top: 80, bottom: 80, left: 80, right: 80 },
              children: [
                new Paragraph({
                  children: [new TextRun({ text: String(cell), font: "Times New Roman", size: 15, bold: r === 0 })],
                  alignment: AlignmentType.CENTER,
                  spacing: { after: 0 },
                }),
              ],
            })
        ),
      })
    ),
  });
}

function image(file, width, height) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 40 },
    children: [
      new ImageRun({
        data: fs.readFileSync(file),
        transformation: { width, height },
      }),
    ],
  });
}

const closedRows = [["Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]];
for (const r of closed) closedRows.push([modelName(r.model), pct(r.accuracy), pct(r.macro_precision), pct(r.macro_recall), pct(r.macro_f1)]);

const openRows = [["Model", "Threshold", "Known Acc", "Unknown Rej.", "FAR", "FRR", "Overall"]];
for (const r of open) openRows.push([modelName(r.model), num(r.threshold), pct(r.known_speaker_accuracy), pct(r.unknown_rejection_accuracy), pct(r.false_acceptance_rate), pct(r.false_rejection_rate), pct(r.overall_open_set_accuracy)]);

const compNames = {
  ecapa_original_float32: "Original",
  ecapa_pca_128: "PCA-128",
  ecapa_pca_64: "PCA-64",
  ecapa_pca_32: "PCA-32",
  ecapa_pca_16: "PCA-16",
  ecapa_float16_centroid: "float16",
  ecapa_int8_centroid: "int8",
};
const compRows = [["Method", "Dim", "Type", "Closed", "Open", "Storage"]];
for (const r of comp) compRows.push([compNames[r.method] ?? r.method, r.embedding_dim, r.centroid_dtype, pct(r.closed_set_accuracy), pct(r.open_set_overall_accuracy), pct(r.storage_ratio_vs_float32)]);

const children = [];
children.push(
  new Paragraph({
    children: [boldRun("SPEAKER IDENTIFICATION ON LIBRISPEECH USING MFCC-GMM, ECAPA-TDNN, AND EMBEDDING COMPRESSION", 24)],
    alignment: AlignmentType.CENTER,
    spacing: { after: 160 },
  })
);
children.push(para("Zheng Jinhui (12311626), Li Yizhang (12212210), Meng Jiaxu (12212317), Shi Yidong (12313636)", { alignment: AlignmentType.CENTER, size: 18, after: 180 }));
children.push(para("Abstract", { bold: true, size: 18, after: 40 }));
children.push(para("This report presents a complete speaker identification system for a speech signal processing course project. Using the LibriSpeech dev-clean subset, we build a 10-speaker closed-set identification task and a small open-set rejection task with five unseen speakers. Three model families are implemented and evaluated: a traditional MFCC statistical-template baseline, one Gaussian Mixture Model per speaker with 4, 8, and 16 components, and a pretrained ECAPA-TDNN speaker embedding model loaded locally without retraining. For open-set evaluation, a model-specific threshold is selected on validation scores and then applied to test utterances. Finally, we add an embedding-level compression experiment for ECAPA using PCA dimensionality reduction and centroid quantization. GMM-16 reaches 90.00% closed-set accuracy among traditional models, while ECAPA reaches 90.00% closed-set accuracy and 88.89% overall open-set accuracy. PCA-128 further improves the compressed ECAPA representation to 91.67% closed-set accuracy and 93.33% open-set accuracy while reducing centroid storage to 66.67% of the original float32 centroid database."));
children.push(
  new Paragraph({
    children: [boldRun("Index Terms-- ", 18), normalRun("speaker identification, MFCC, Gaussian Mixture Model, ECAPA-TDNN, open-set rejection, embedding compression, PCA, quantization.")],
    spacing: { after: 140 },
  })
);

const sections = [
  ["1. Introduction", [
    "Speaker identification aims to determine which enrolled speaker produced a given speech utterance. In the closed-set setting, every test utterance is assumed to belong to one of the registered speakers. In real applications, however, an input may come from an unseen speaker; therefore, an open-set system must be able to reject unknown speakers instead of forcing every utterance into a registered identity.",
    "This project focuses on a practical and reproducible pipeline. We first implement traditional MFCC-based methods, including a simple statistical-template baseline and speaker-dependent GMMs. We then compare them with a pretrained ECAPA-TDNN speaker embedding model. The final part studies whether ECAPA speaker representations can be compressed at the embedding or centroid level without modifying the neural network itself.",
  ]],
  ["2. Dataset and Split", [
    "The experiments use the LibriSpeech dev-clean subset. Speaker IDs are selected automatically according to available utterance counts. Ten speakers are registered, and each registered speaker contributes 40 utterances: 28 for training or enrollment, 6 for validation, and 6 for testing. Five additional speakers are selected as unknown speakers, each contributing 6 validation and 6 test utterances for open-set rejection.",
    "The final registered speaker IDs are 1988, 2277, 2412, 2428, 5895, 6319, 6345, 777, 7850, and 7976. The unknown speaker IDs are 1993, 2803, 3853, 5694, and 6295. The resulting split contains 280 training utterances, 60 validation utterances, 60 closed-set test utterances, 30 unknown validation utterances, and 30 unknown test utterances.",
  ]],
  ["3. Feature Extraction and Traditional Models", [
    "All traditional models use MFCC features. For each utterance, 13 MFCC coefficients are extracted together with delta and delta-delta coefficients. Cepstral normalization is applied at the utterance level. The baseline model converts frame-level MFCC features into an utterance-level embedding by concatenating the mean and standard deviation over time. A speaker template is then computed by averaging the enrollment embeddings of each speaker, and prediction is made using cosine similarity.",
    "The main traditional model is a one-GMM-per-speaker system. For each registered speaker, all training frames are pooled and used to train a diagonal-covariance Gaussian Mixture Model. We evaluate 4, 8, and 16 components. During inference, the average frame log-likelihood is computed for each speaker model, and the speaker with the highest score is selected.",
  ]],
  ["4. ECAPA-TDNN Speaker Embedding Model", [
    "The ECAPA-TDNN model is used as a pretrained embedding extractor. It is loaded locally from the project checkpoint directory and is not trained or pruned in this project. Each utterance is mapped to a 192-dimensional speaker embedding. For each registered speaker, a centroid embedding is computed from that speaker's enrollment utterances. Closed-set prediction is performed by cosine similarity between a test embedding and all speaker centroids.",
    "This design keeps the neural speaker representation fixed and shifts the course project focus to system construction, comparison against traditional signal-processing baselines, open-set thresholding, and representation-level compression.",
  ]],
  ["5. Open-set Rejection", [
    "Open-set rejection is implemented by thresholding the maximum speaker score. For a test utterance, if the maximum score over registered speakers is below a model-specific threshold, the system outputs Unknown; otherwise, it outputs the registered speaker with the highest score. The threshold is selected on validation data using registered validation utterances and unknown validation utterances. Test utterances are not used when selecting the threshold.",
    "We report known-speaker accuracy, unknown rejection accuracy, false acceptance rate (FAR), false rejection rate (FRR), and overall open-set accuracy. Overall open-set accuracy combines correct known-speaker identification and correct rejection of unknown speakers. This metric is different from closed-set accuracy and should not be mixed with it.",
  ]],
  ["6. Embedding Compression and Quantization", [
    "The innovation experiment studies whether ECAPA embeddings and speaker centroids can be compressed without modifying the pretrained ECAPA network. This is safer than pruning the neural network because the checkpoint and inference architecture remain unchanged. Compression is applied only after the ECAPA embedding has been extracted.",
    "Scheme A uses PCA dimensionality reduction. PCA is fitted only on enrollment embeddings, then applied to training, validation, test, and unknown embeddings. After projection, speaker centroids are recomputed in the lower-dimensional PCA space, and cosine similarity is used for scoring. We evaluate the original 192-dimensional embedding and PCA dimensions of 128, 64, 32, and 16.",
    "Scheme B compresses the speaker centroid database. Float16 centroid compression stores centroids in half precision and casts them back to float32 during scoring. Int8 centroid quantization uses symmetric per-centroid quantization with one scale value per speaker centroid. This reduces storage while leaving the original ECAPA embedding extractor unchanged.",
  ]],
];
for (const [h, ps] of sections) {
  children.push(heading(h));
  for (const t of ps) children.push(para(t));
}

children.push(heading("7. Results and Discussion"));
children.push(caption("Table 1. Closed-set speaker identification results."));
children.push(table(closedRows));
children.push(para("The baseline performs close to random guessing for a 10-speaker task. The GMM models improve substantially as the number of mixture components increases. GMM-16 reaches 90.00% closed-set accuracy and is the best traditional model. ECAPA also reaches 90.00% accuracy and obtains the highest macro F1 score, showing that pretrained speaker embeddings are highly competitive even without task-specific training."));
children.push(image(path.join(ROOT, "results/figures/model_comparison_accuracy.png"), 250, 130));
children.push(caption("Fig. 1. Closed-set accuracy comparison across baseline, GMM, and ECAPA models."));
children.push(caption("Table 2. Open-set rejection results."));
children.push(table(openRows));
children.push(para("Open-set evaluation reveals a larger gap between traditional statistical modeling and pretrained speaker embeddings. GMM-16 improves over smaller GMMs, but it still rejects only 40.00% of unknown-speaker test utterances. ECAPA achieves 88.89% overall open-set accuracy and 86.67% unknown rejection accuracy, indicating that pretrained embeddings provide a more discriminative score space for separating enrolled and unseen speakers."));
children.push(image(path.join(ROOT, "results/figures/open_set_score_distribution.png"), 250, 150));
children.push(caption("Fig. 2. Open-set validation score distribution used for threshold selection."));
children.push(caption("Table 3. ECAPA embedding compression and centroid quantization results."));
children.push(table(compRows));
children.push(para("PCA-128 gives the best compressed result, reaching 91.67% closed-set accuracy and 93.33% open-set accuracy while reducing centroid storage to 66.67% of the original float32 centroid database. PCA-64 is a stronger storage-accuracy compromise, preserving 92.22% open-set accuracy with only 33.33% centroid storage. Float16 and int8 centroid compression preserve the original ECAPA closed-set and open-set accuracy on this split, while int8 reduces centroid storage to 25.52%. These results suggest that embedding-level compression is a low-risk way to make the speaker database lighter without retraining ECAPA."));
children.push(image(path.join(ROOT, "results/figures/embedding_compression_storage_vs_accuracy.png"), 250, 190));
children.push(caption("Fig. 3. Storage versus open-set accuracy trade-off for ECAPA embedding compression."));

children.push(heading("8. Conclusion"));
children.push(para("This project implements a complete speaker identification system with both traditional signal-processing models and a pretrained deep speaker embedding model. The MFCC template baseline is simple but weak. GMMs provide a strong traditional baseline, with GMM-16 reaching 90.00% closed-set accuracy. ECAPA matches the best closed-set accuracy and performs much better in open-set rejection. The compression experiment further shows that ECAPA representations can be reduced or quantized at the embedding/centroid level while maintaining strong performance. Overall, the final system demonstrates the progression from handcrafted acoustic features to probabilistic modeling, pretrained speaker embeddings, open-set thresholding, and lightweight representation compression."));

children.push(heading("References"));
[
  '[1] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, "LibriSpeech: An ASR corpus based on public domain audio books," in Proc. ICASSP, 2015.',
  '[2] D. A. Reynolds and R. C. Rose, "Robust text-independent speaker identification using Gaussian mixture speaker models," IEEE Transactions on Speech and Audio Processing, 1995.',
  '[3] B. Desplanques, J. Thienpondt, and K. Demuynck, "ECAPA-TDNN: Emphasized channel attention, propagation and aggregation in TDNN based speaker verification," in Proc. Interspeech, 2020.',
  '[4] M. Ravanelli et al., "SpeechBrain: A general-purpose speech toolkit," arXiv:2106.04624, 2021.',
  '[5] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," Journal of Machine Learning Research, 2011.',
].forEach((r) => children.push(para(r, { size: 16, after: 40 })));

const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: "Times New Roman", size: 18 },
        paragraph: { spacing: { line: 220, after: 80 } },
      },
    },
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 11907, height: 16840 },
          margin: { top: 1418, right: 907, bottom: 1418, left: 907 },
        },
      },
      children,
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(OUT, buffer);
  console.log(`Wrote ${OUT} (${buffer.length} bytes)`);
});
