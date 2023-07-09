const tf = require("@tensorflow/tfjs");
require("@tensorflow/tfjs");
const use = require("@tensorflow-models/universal-sentence-encoder");
import { UMAP } from "umap-js";
// import {
//   SpellCheck as nlpSpellCheck,
//   NGrams,
//   SentimentAnalyzer as nlpSentimentAnalyzer,
//   SentimentManager, // to handle multiple languages at the same time
// } from "node-nlp";
import {
  // Spellcheck as naturalSpellCheck,
  SentimentAnalyzer as naturalSentimentAnalyzer,
  PorterStemmer, // use AFINN lexicon for sentiment analysis
} from "natural";
// import dictionaryEnCa from "dictionary-en-ca";
import nspell from "nspell";
import * as $ from "jquery";
import Chart from "chart.js/auto"; // https://stackoverflow.com/a/67143648

let aff;
let dic;
let spell;
loadDictionary();
async function loadDictionary() {
  aff = await fetch("./index.aff").then((response) => {
    return response.text();
  });
  dic = await fetch("./index.dic").then((response) => {
    return response.text();
  });
  spell = nspell(aff, dic);
  console.log("loaded dictionary");
  checkTypos();
}

let inputsPrev = $("#inputs").val();
$("#inputs").on("keyup", (event) => {
  if (inputsPrev === $("#inputs").val()) return;
  checkTypos();
  inputsPrev = $("#inputs").val();
});
$("body").on("click", ".replace-typo", (event) => {
  const button = $(event.target);
  const word = button.data("word");
  const suggestion = button.text();
  const yes = confirm(
    `Do you want to replace *ALL* instances of "${word}" with "${suggestion}"? Otherwise consider manually replacing individual cases.`
  );
  if (yes) {
    $("#inputs").val($("#inputs").val().replaceAll(word, suggestion));
    checkTypos();
  }
});
function checkTypos() {
  if (!spell) return;
  const words = Array.from(
    new Set(
      $("#inputs")
        .val()
        // .replace(/[.,\/#!$%\^&\*;:{}=\_`~()!?]/g, "") // allow ' and - in words
        // .split(/\s/)
        .split(/[\s.,\/#!$%\^&\*;:{}=\_`~()!?]/) // allow ' and - in words
        .filter(Boolean)
    )
  );
  const suggestions = words
    .map((w) => {
      return {
        word: w,
        suggestions: spell.suggest(w),
      };
    })
    .filter((w) => w.suggestions.length);
  const html = suggestions.map(
    (w) =>
      `<p class="typo-row"><span class="red">${
        w.word
      }</span> <span>&rarr;</span>
        ${w.suggestions
          .map(
            (suggestion) =>
              `<button class="replace-typo" data-word="${w.word}">${suggestion}</button>`
          )
          .join("")}
      </p>`
  );
  $("#typo_fix_suggestions").html(html);
  $("#suggestions").toggleClass("d-none", !suggestions.length);
}

let chart;

$("#start").on("click", () => {
  const sentences = $("#inputs").val().split("\n").filter(Boolean);
  if (sentences?.length) {
    $("#start").prop("disabled", true);
    $(".chartjs-tooltip").remove();
    runAnalysis(sentences, () => {
      $("#start").prop("disabled", false);
      $("#start").text("Re-run");
    });
  }
});

async function runAnalysis(sentences, callback) {
  if (chart) chart.destroy();

  $("#chart").addClass("in-progress");
  $("#status").css("color", "red");

  showStatus("Creating model...");
  const model = await use.load();

  showStatus("Creating embeddings...");
  const embeddings = await model.embed(sentences);
  const sentenceEmbeddingsAsArray = [];
  sentences.forEach((sentence, i) => {
    const sentenceEmbedding = tf.slice(embeddings, [i, 0], [1]);
    const sentenceEmbeddingAsArray = sentenceEmbedding.dataSync();
    sentenceEmbeddingsAsArray.push(sentenceEmbeddingAsArray);
  });

  showStatus("Creating plottable data with UMAP...");
  const dimensions = 2; // 2 = 2D
  const numberOfNeighbours = 3; // Math.min(15, Math.max(3, Math.ceil(sentenceEmbeddingsAsArray.length / 2)));
  console.log("numberOfNeighbours", numberOfNeighbours);
  const umap = new UMAP({
    //nEpochs: 100, // nEpochs is computed automatically by default
    nComponents: dimensions,
    nNeighbors: numberOfNeighbours,
    minDist: 0.1, // default: 0.1
    spread: 1.0, // default: 1.0
    // other parameters: https://github.com/PAIR-code/umap-js/#parameters
  });
  const plottableData = umap.fit(sentenceEmbeddingsAsArray);

  showStatus("Plotting data...");
  chart = plot(plottableData, sentences, () => {
    $("#chart").removeClass("in-progress");
    $("#status").css("color", "green");

    showStatus(
      "Visualization ready! Hover over points to read comments. The Universal-Sentence-Encoder model generates embeddings, then the code tries to group semantically similar comments near each other using UMAP (as opposed to PCA or t-SNE). The UMAP algorithm is stochastic (uses randomness) to speed up dimension reduction. More info at https://github.com/hchiam/learning-tfjs-umap"
    );

    if (callback) callback();
  });
}

function showStatus(message) {
  $("#status").text(message);
  console.log(message);
}

function plot(coordinatesArray, labels, callback) {
  const data = coordinatesArray.map((x) => {
    return { x: x[0], y: x[1] };
  });

  console.log("labels", labels);
  console.log("data", data);

  const chart = new Chart("chart", {
    type: "scatter",
    data: {
      labels: labels,
      datasets: [
        {
          data: data,
          pointBackgroundColor: "black",
          pointRadius: 7,
        },
      ],
    },
    options: {
      aspectRatio: 1,
      maintainAspectRatio: true,
      // interaction: {
      //   mode: "nearest",
      // },
      plugins: {
        legend: {
          display: false,
        },
        tooltip: {
          // intersect: false,

          enabled: false,

          external: function (context) {
            const canvasBox = context.chart.canvas.getBoundingClientRect();
            const tooltip = context.tooltip;
            const title = String(tooltip.title);
            const left = String(tooltip.caretX + canvasBox.left) + "px";
            const top = String(tooltip.caretY + canvasBox.top) + "px";

            const tooltipSelector = `.chartjs-tooltip[data-title="${title}"]`;
            const alreadyHaveTooltip = $(tooltipSelector).length > 0;

            if (alreadyHaveTooltip) return;

            const tooltipEl = document.createElement("div");
            tooltipEl.className = "chartjs-tooltip";
            tooltipEl.innerText = title;
            tooltipEl.dataset.title = title;
            tooltipEl.style.left = left;
            tooltipEl.dataset.left = left;
            tooltipEl.style.top = top;
            tooltipEl.dataset.top = top;
            tooltipEl.style.background = "#ffffff80";
            tooltipEl.style.position = "absolute";
            tooltipEl.style.pointerEvents = "none";
            tooltipEl.style.borderRadius = "0.3rem";
            tooltipEl.style.padding = "0.1rem";
            document.body.appendChild(tooltipEl);
            // setTimeout(() => $(tooltipSelector).remove(), 3000);

            $("#hide_tooltips").prop("disabled", false);
          },
        },
      },
      responsive: false,
      scales: {
        x: {
          ticks: {
            display: false,
          },
          grid: {
            display: false,
          },
        },
        y: {
          ticks: {
            display: false,
          },
          grid: {
            display: false,
          },
        },
      },
    },
  });

  $(window).on("resize", () => {
    $(".chartjs-tooltip").remove();
  });

  $("#hide_tooltips").on("click", () => {
    $(".chartjs-tooltip").remove();
  });

  if (callback) callback();

  return chart;
}
