$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ben04\Desktop\speaker_id_project"
$Template = "C:\Users\ben04\Downloads\WordTemplate(1) (1).docx"
$Output = Join-Path $ProjectRoot "Speaker_ID_Project_Report.docx"
$PdfOutput = Join-Path $ProjectRoot "outputs\report_render_word\Speaker_ID_Project_Report.pdf"

function Pct([double]$x) { return "{0:P2}" -f $x }
function Num4([double]$x) { return "{0:N4}" -f $x }

$closed = Import-Csv (Join-Path $ProjectRoot "results\metrics\closed_set_metrics.csv")
$open = Import-Csv (Join-Path $ProjectRoot "results\metrics\open_set_metrics.csv")
$compression = Import-Csv (Join-Path $ProjectRoot "results\metrics\embedding_compression_metrics.csv")

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$doc = $word.Documents.Open($Template)
$doc.Content.Delete()

# A4 two-column IEEE-style layout.
$doc.PageSetup.PaperSize = 7
$doc.PageSetup.TopMargin = $word.CentimetersToPoints(2.5)
$doc.PageSetup.BottomMargin = $word.CentimetersToPoints(2.5)
$doc.PageSetup.LeftMargin = $word.CentimetersToPoints(1.6)
$doc.PageSetup.RightMargin = $word.CentimetersToPoints(1.6)
$doc.PageSetup.TextColumns.SetCount(2)

$sel = $word.Selection

function Add-Para([string]$Text, [string]$StyleName = "", [int]$FontSize = 9, [bool]$Bold = $false, [int]$Align = 0) {
    $script:sel.EndKey(6) | Out-Null
    $script:sel.TypeText($Text)
    $script:sel.TypeParagraph()
    $p = $script:doc.Paragraphs.Item($script:doc.Paragraphs.Count - 1)
    if ($StyleName -ne "") {
        try { $p.Range.Style = $script:doc.Styles.Item($StyleName) } catch {}
    }
    $p.Range.Font.Name = "Times New Roman"
    $p.Range.Font.Size = $FontSize
    $p.Range.Font.Bold = [int]$Bold
    $p.Alignment = $Align
    $p.SpaceAfter = 3
}

function Add-Heading([string]$Text) {
    Add-Para $Text.ToUpper() "heading 2" 10 $true 1
}

function Add-SubHeading([string]$Text) {
    Add-Para $Text "heading 3" 9 $true 0
}

function Add-Table($Rows, [string]$Caption) {
    Add-Para $Caption "" 8 $false 1
    $rowCount = $Rows.Count
    $colCount = $Rows[0].Count
    $range = $script:sel.Range
    $table = $script:doc.Tables.Add($range, $rowCount, $colCount)
    $table.Range.Font.Name = "Times New Roman"
    $table.Range.Font.Size = 7.5
    $table.Borders.Enable = 1
    $table.Rows.Item(1).Range.Bold = 1
    $table.Rows.Item(1).Shading.BackgroundPatternColor = 15189684
    for ($r = 1; $r -le $rowCount; $r++) {
        for ($c = 1; $c -le $colCount; $c++) {
            $table.Cell($r, $c).Range.Text = [string]$Rows[$r-1][$c-1]
            $table.Cell($r, $c).Range.ParagraphFormat.Alignment = 1
            $table.Cell($r, $c).VerticalAlignment = 1
        }
    }
    $table.AutoFitBehavior(1)
    $script:sel.SetRange($table.Range.End, $table.Range.End)
    $script:sel.TypeParagraph()
}

function Add-Figure([string]$Path, [string]$Caption, [double]$WidthInches = 3.2) {
    $script:sel.EndKey(6) | Out-Null
    $shape = $script:sel.InlineShapes.AddPicture($Path, $false, $true)
    $shape.LockAspectRatio = -1
    $shape.Width = $WidthInches * 72
    $script:sel.TypeParagraph()
    Add-Para $Caption "" 8 $false 1
}

Add-Para "SPEAKER IDENTIFICATION ON LIBRISPEECH USING MFCC-GMM, ECAPA-TDNN, AND EMBEDDING COMPRESSION" "heading 1" 12 $true 1
Add-Para "Zheng Jinhui (12311626), Li Yizhang (12212210), Meng Jiaxu (12212317), Shi Yidong (12313636)" "Author" 9 $false 1
Add-Para "Abstract" "Abstract" 9 $true 0
Add-Para "This report presents a complete speaker identification system for a speech signal processing course project. Using the LibriSpeech dev-clean subset, we build a 10-speaker closed-set identification task and a small open-set rejection task with five unseen speakers. Three model families are implemented and evaluated: a traditional MFCC statistical-template baseline, one Gaussian Mixture Model per speaker with 4, 8, and 16 components, and a pretrained ECAPA-TDNN speaker embedding model loaded locally without retraining. For open-set evaluation, a model-specific threshold is selected on validation scores and then applied to test utterances. Finally, we add an embedding-level compression experiment for ECAPA using PCA dimensionality reduction and centroid quantization. GMM-16 reaches 90.00% closed-set accuracy among traditional models, while ECAPA reaches 90.00% closed-set accuracy and 88.89% overall open-set accuracy. PCA-128 further improves the compressed ECAPA representation to 91.67% closed-set accuracy and 93.33% open-set accuracy while reducing centroid storage to 66.67% of the original float32 centroid database." "" 9 $false 0
Add-Para "Index Terms-- speaker identification, MFCC, Gaussian Mixture Model, ECAPA-TDNN, open-set rejection, embedding compression, PCA, quantization." "Body Text Indent" 9 $false 0

Add-Heading "1. Introduction"
Add-Para "Speaker identification aims to determine which enrolled speaker produced a given speech utterance. In the closed-set setting, every test utterance is assumed to belong to one of the registered speakers. In real applications, however, an input may come from an unseen speaker; therefore, an open-set system must be able to reject unknown speakers instead of forcing every utterance into a registered identity." "" 9 $false 0
Add-Para "This project focuses on a practical and reproducible pipeline. We first implement traditional MFCC-based methods, including a simple statistical-template baseline and speaker-dependent GMMs. We then compare them with a pretrained ECAPA-TDNN speaker embedding model. The final part studies whether ECAPA speaker representations can be compressed at the embedding or centroid level without modifying the neural network itself." "" 9 $false 0

Add-Heading "2. Dataset and Split"
Add-Para "The experiments use the LibriSpeech dev-clean subset. Speaker IDs are selected automatically according to available utterance counts. Ten speakers are registered, and each registered speaker contributes 40 utterances: 28 for training or enrollment, 6 for validation, and 6 for testing. Five additional speakers are selected as unknown speakers, each contributing 6 validation and 6 test utterances for open-set rejection." "" 9 $false 0
Add-Para "The final registered speaker IDs are 1988, 2277, 2412, 2428, 5895, 6319, 6345, 777, 7850, and 7976. The unknown speaker IDs are 1993, 2803, 3853, 5694, and 6295. The resulting split contains 280 training utterances, 60 validation utterances, 60 closed-set test utterances, 30 unknown validation utterances, and 30 unknown test utterances." "" 9 $false 0

Add-Heading "3. Feature Extraction and Traditional Models"
Add-Para "All traditional models use MFCC features. For each utterance, 13 MFCC coefficients are extracted together with delta and delta-delta coefficients. Cepstral normalization is applied at the utterance level. The baseline model converts frame-level MFCC features into an utterance-level embedding by concatenating the mean and standard deviation over time. A speaker template is then computed by averaging the enrollment embeddings of each speaker, and prediction is made using cosine similarity." "" 9 $false 0
Add-Para "The main traditional model is a one-GMM-per-speaker system. For each registered speaker, all training frames are pooled and used to train a diagonal-covariance Gaussian Mixture Model. We evaluate 4, 8, and 16 components. During inference, the average frame log-likelihood is computed for each speaker model, and the speaker with the highest score is selected." "" 9 $false 0

Add-Heading "4. ECAPA-TDNN Speaker Embedding Model"
Add-Para "The ECAPA-TDNN model is used as a pretrained embedding extractor. It is loaded locally from the project checkpoint directory and is not trained or pruned in this project. Each utterance is mapped to a 192-dimensional speaker embedding. For each registered speaker, a centroid embedding is computed from that speaker's enrollment utterances. Closed-set prediction is performed by cosine similarity between a test embedding and all speaker centroids." "" 9 $false 0
Add-Para "This design keeps the neural speaker representation fixed and shifts the course project focus to system construction, comparison against traditional signal-processing baselines, open-set thresholding, and representation-level compression." "" 9 $false 0

Add-Heading "5. Open-set Rejection"
Add-Para "Open-set rejection is implemented by thresholding the maximum speaker score. For a test utterance, if the maximum score over registered speakers is below a model-specific threshold, the system outputs Unknown; otherwise, it outputs the registered speaker with the highest score. The threshold is selected on validation data using registered validation utterances and unknown validation utterances. Test utterances are not used when selecting the threshold." "" 9 $false 0
Add-Para "We report known-speaker accuracy, unknown rejection accuracy, false acceptance rate (FAR), false rejection rate (FRR), and overall open-set accuracy. Overall open-set accuracy combines correct known-speaker identification and correct rejection of unknown speakers. This metric is different from closed-set accuracy and should not be mixed with it." "" 9 $false 0

Add-Heading "6. Embedding Compression and Quantization"
Add-Para "The innovation experiment studies whether ECAPA embeddings and speaker centroids can be compressed without modifying the pretrained ECAPA network. This is safer than pruning the neural network because the checkpoint and inference architecture remain unchanged. Compression is applied only after the ECAPA embedding has been extracted." "" 9 $false 0
Add-Para "Scheme A uses PCA dimensionality reduction. PCA is fitted only on enrollment embeddings, then applied to training, validation, test, and unknown embeddings. After projection, speaker centroids are recomputed in the lower-dimensional PCA space, and cosine similarity is used for scoring. We evaluate the original 192-dimensional embedding and PCA dimensions of 128, 64, 32, and 16." "" 9 $false 0
Add-Para "Scheme B compresses the speaker centroid database. Float16 centroid compression stores centroids in half precision and casts them back to float32 during scoring. Int8 centroid quantization uses symmetric per-centroid quantization with one scale value per speaker centroid. This reduces storage while leaving the original ECAPA embedding extractor unchanged." "" 9 $false 0

Add-Heading "7. Results and Discussion"
$closedRows = @()
$closedRows += ,@("Model", "Accuracy", "Macro Prec.", "Macro Rec.", "Macro F1")
foreach ($r in $closed) {
    $name = $r.model.Replace("baseline","Baseline").Replace("gmm_","GMM-").Replace("ecapa","ECAPA")
    $closedRows += ,@($name, (Pct $r.accuracy), (Pct $r.macro_precision), (Pct $r.macro_recall), (Pct $r.macro_f1))
}
Add-Table $closedRows "Table 1. Closed-set speaker identification results."
Add-Para "The baseline performs close to random guessing for a 10-speaker task. The GMM models improve substantially as the number of mixture components increases. GMM-16 reaches 90.00% closed-set accuracy and is the best traditional model. ECAPA also reaches 90.00% accuracy and obtains the highest macro F1 score, showing that pretrained speaker embeddings are highly competitive even without task-specific training." "" 9 $false 0
Add-Figure (Join-Path $ProjectRoot "results\figures\model_comparison_accuracy.png") "Fig. 1. Closed-set accuracy comparison across baseline, GMM, and ECAPA models." 3.25

$openRows = @()
$openRows += ,@("Model", "Threshold", "Known", "Unknown Rej.", "FAR", "FRR", "Overall")
foreach ($r in $open) {
    $name = $r.model.Replace("baseline","Baseline").Replace("gmm_","GMM-").Replace("ecapa","ECAPA")
    $openRows += ,@($name, (Num4 $r.threshold), (Pct $r.known_speaker_accuracy), (Pct $r.unknown_rejection_accuracy), (Pct $r.false_acceptance_rate), (Pct $r.false_rejection_rate), (Pct $r.overall_open_set_accuracy))
}
Add-Table $openRows "Table 2. Open-set rejection results."
Add-Para "Open-set evaluation reveals a larger gap between traditional statistical modeling and pretrained speaker embeddings. GMM-16 improves over smaller GMMs, but it still rejects only 40.00% of unknown-speaker test utterances. ECAPA achieves 88.89% overall open-set accuracy and 86.67% unknown rejection accuracy, indicating that pretrained embeddings provide a more discriminative score space for separating enrolled and unseen speakers." "" 9 $false 0
Add-Figure (Join-Path $ProjectRoot "results\figures\open_set_score_distribution.png") "Fig. 2. Open-set validation score distribution used for threshold selection." 3.25

$compRows = @()
$compRows += ,@("Method", "Dim", "Type", "Closed", "Open", "Storage")
foreach ($r in $compression) {
    $name = switch ($r.method) {
        "ecapa_original_float32" {"Original"}
        "ecapa_pca_128" {"PCA-128"}
        "ecapa_pca_64" {"PCA-64"}
        "ecapa_pca_32" {"PCA-32"}
        "ecapa_pca_16" {"PCA-16"}
        "ecapa_float16_centroid" {"float16"}
        "ecapa_int8_centroid" {"int8"}
        default {$r.method}
    }
    $compRows += ,@($name, $r.embedding_dim, $r.centroid_dtype, (Pct $r.closed_set_accuracy), (Pct $r.open_set_overall_accuracy), (Pct $r.storage_ratio_vs_float32))
}
Add-Table $compRows "Table 3. ECAPA embedding compression and centroid quantization results."
Add-Para "PCA-128 gives the best compressed result, reaching 91.67% closed-set accuracy and 93.33% open-set accuracy while reducing centroid storage to 66.67% of the original float32 centroid database. PCA-64 is a stronger storage-accuracy compromise, preserving 92.22% open-set accuracy with only 33.33% centroid storage. Float16 and int8 centroid compression preserve the original ECAPA closed-set and open-set accuracy on this split, while int8 reduces centroid storage to 25.52%. These results suggest that embedding-level compression is a low-risk way to make the speaker database lighter without retraining ECAPA." "" 9 $false 0
Add-Figure (Join-Path $ProjectRoot "results\figures\embedding_compression_storage_vs_accuracy.png") "Fig. 3. Storage versus open-set accuracy trade-off for ECAPA embedding compression." 3.25

Add-Heading "8. Conclusion"
Add-Para "This project implements a complete speaker identification system with both traditional signal-processing models and a pretrained deep speaker embedding model. The MFCC template baseline is simple but weak. GMMs provide a strong traditional baseline, with GMM-16 reaching 90.00% closed-set accuracy. ECAPA matches the best closed-set accuracy and performs much better in open-set rejection. The compression experiment further shows that ECAPA representations can be reduced or quantized at the embedding/centroid level while maintaining strong performance. Overall, the final system demonstrates the progression from handcrafted acoustic features to probabilistic modeling, pretrained speaker embeddings, open-set thresholding, and lightweight representation compression." "" 9 $false 0

Add-Heading "References"
$refs = @(
    "[1] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, `"LibriSpeech: An ASR corpus based on public domain audio books,`" in Proc. ICASSP, 2015.",
    "[2] D. A. Reynolds and R. C. Rose, `"Robust text-independent speaker identification using Gaussian mixture speaker models,`" IEEE Transactions on Speech and Audio Processing, 1995.",
    "[3] B. Desplanques, J. Thienpondt, and K. Demuynck, `"ECAPA-TDNN: Emphasized channel attention, propagation and aggregation in TDNN based speaker verification,`" in Proc. Interspeech, 2020.",
    "[4] M. Ravanelli et al., `"SpeechBrain: A general-purpose speech toolkit,`" arXiv:2106.04624, 2021.",
    "[5] F. Pedregosa et al., `"Scikit-learn: Machine learning in Python,`" Journal of Machine Learning Research, 2011."
)
foreach ($ref in $refs) { Add-Para $ref "Reference" 8 $false 0 }

$doc.SaveAs2($Output)
New-Item -ItemType Directory -Force -Path (Split-Path $PdfOutput) | Out-Null
$doc.ExportAsFixedFormat($PdfOutput, 17)
$pages = $doc.ComputeStatistics(2)
$doc.Close($false)
$word.Quit()

Write-Output "DOCX=$Output"
Write-Output "PDF=$PdfOutput"
Write-Output "Pages=$pages"
