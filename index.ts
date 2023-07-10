const tf = require("@tensorflow/tfjs");
require("@tensorflow/tfjs");
const use = require("@tensorflow-models/universal-sentence-encoder");
import { UMAP } from "umap-js";
const knnClassifier = require("@tensorflow-models/knn-classifier");
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

const stemmer = PorterStemmer;
const analyzer = new naturalSentimentAnalyzer("English", stemmer, "afinn");

let aff;
let dic;
let spell;
loadDictionary();

let classColourIndex = 0;
const classColourChoices = [
  "brown",
  "red",
  "orange",
  "yellow",
  "green",
  "blue",
  "purple",
  "grey",
  "white",
];
const knn = knnClassifier.create();

showSentiments();

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
$("body").on("click", ".replace-typo, .replace-typo-other", function (event) {
  const button = $(this);
  const isOther = button.hasClass("replace-typo-other");
  const word = button.data("word");
  const suggestion = isOther ? button.next("input").val() : button.text();
  if (!suggestion) {
    button.next("input").focus();
    return;
  }
  const yes = confirm(
    `Do you want to replace *ALL* instances of "${word}" with "${suggestion}"? Otherwise consider manually replacing individual cases.`
  );
  if (yes) {
    const wholeWords = new RegExp("\\b" + word + "\\b", "g");
    $("#inputs").val($("#inputs").val().replace(wholeWords, suggestion));
    checkTypos();
  }
});

function checkTypos() {
  if (!spell) return;
  const words = Array.from(new Set(getWordsFromInputs()));
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
          <button class="replace-typo-other" data-word="${
            w.word
          }">other:</button>
          <input "replace-typo-other" placeholder="${w.word}">
      </p>`
  );
  $("#typo_fix_suggestions").html(html);
  $("#suggestions").toggleClass("d-none", !suggestions.length);
  showSentiments();
}

function getSentencesFromInputs() {
  return $("#inputs").val().split("\n").filter(Boolean);
}

function getWordsFromInputs() {
  return getWordsFromSentence($("#inputs").val());
}

function getWordsFromSentence(sentence) {
  // allow ' and - in words
  // split on punctuation, including "
  return (
    sentence // .replace(/[.,\/#!$%\^&\*;:{}=\_`~()!?"]/g, "")
      // .split(/\s/)
      .split(/[\s.,\/#!$%\^&\*;:{}=\_`~()!?"]/)
      .filter(Boolean)
  );
}

function showSentiments() {
  const sentences = getSentencesFromInputs();
  const sentiments = sentences
    .map((sentence) => {
      const score = analyzer.getSentiment(getWordsFromSentence(sentence));
      return {
        sentence: sentence,
        score: Math.round(score * 10) / 10,
      };
    })
    .map((s) => {
      const score = s.score;
      let labelSymbol = "😐";
      let label = "neutral";
      if (score > 0) {
        labelSymbol = "✅";
        label = "positive";
      } else if (score < 0) {
        labelSymbol = "❌";
        label = "negative";
      }
      return `${labelSymbol} likely ${label}: ${score} : "${s.sentence}"`;
    });
  $("#sentiments").find("textarea").val(sentiments.join("\n"));
  $("#sentiments").toggleClass("d-none", !sentiments.length);
}

let chart;

$("#start").on("click", () => {
  const sentences = getSentencesFromInputs();
  if (sentences?.length) {
    $("#similarities").find("button, input").prop("disabled", true);
    $(".chartjs-tooltip").remove();
    runAnalysis(sentences, () => {
      $("#similarities").find("button, input").prop("disabled", false);
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
  const numberOfNeighbours = $("#nNeighbors").val() || 3; // Math.min(15, Math.max(3, Math.ceil(sentenceEmbeddingsAsArray.length / 2)));
  const minDist = $("#minDist").val() || 0.1;
  const umap = new UMAP({
    //nEpochs: 100, // nEpochs is computed automatically by default
    nComponents: dimensions,
    nNeighbors: numberOfNeighbours,
    minDist: minDist, // default: 0.1
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

  await processKnn(sentences, plottableData);
}

function showStatus(message) {
  $("#status").text(message);
  console.log(message);
}

function plot(coordinatesArray, labels, callback) {
  classColourIndex = 0;

  const data = coordinatesArray.map((x) => {
    return { x: x[0], y: x[1] };
  });

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
            const canvasBox = $("#chart").position();
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
      onClick: (event, elements, chart) => {
        const dataset = chart.data.datasets[0];

        let colour = classColourChoices[classColourIndex];
        if (classColourIndex < classColourChoices.length) {
          classColourIndex++;
        } else {
          colour = randomColour();
        }

        const notClickedYet = !Array.isArray(dataset["pointBackgroundColor"]);
        if (notClickedYet) {
          dataset["pointBackgroundColor"] = dataset.data.map((v, i) =>
            i == elements[0]?.index ? colour : "black"
          );
        } else if (elements[0]?.index) {
          dataset["pointBackgroundColor"][elements[0].index] = colour;
        }

        chart.update();
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

function randomColour() {
  const result = [];
  const allowedCharacters = "0123456789abcdef";
  const numberOfOptions = allowedCharacters.length;
  for (let i = 0; i < 6; i++) {
    result.push(
      allowedCharacters.charAt(Math.floor(Math.random() * numberOfOptions))
    );
  }
  return "#" + result.join("");
}

async function processKnn(sentences, plottableData) {
  $("#groups").find("textarea").val("");
  knn.clearAllClasses();
  const numberOfClasses = $("#numberOfClasses").val() || 3;
  const kNearestNeighbours = $("#kNearestNeighbours").val() || 1; // Math.floor(Math.sqrt(plottableData.length)) || 1;
  const usedExamples = Object.create(null);
  for (let example = 0; example < numberOfClasses; example++) {
    let randomIndex = Math.floor(Math.random() * plottableData.length);
    while (randomIndex in usedExamples) {
      randomIndex = Math.floor(Math.random() * plottableData.length);
    }
    usedExamples[randomIndex] = true;
    const x = plottableData[randomIndex][0];
    const y = plottableData[randomIndex][1];
    knn.addExample(tf.tensor([x, y]), example);
  }
  const classified = await Promise.all(
    sentences.map(async (s, i) => {
      const x = plottableData[i][0];
      const y = plottableData[i][1];
      const result = await knn.predictClass(
        tf.tensor([x, y]),
        kNearestNeighbours
      );
      // console.log("result", result.classIndex);
      return { sentence: s, class: result.classIndex };
    })
  );

  classified.sort((a, b) => a.class - b.class);

  let previousClass = 0;
  const val = classified
    .map((s) => {
      const newLine = s.class !== previousClass ? "\n" : "";
      previousClass = s.class;
      return `${newLine}${s.class + 1} ${s.sentence}`;
    })
    .join("\n");

  $("#groups").find("textarea").val(val);
}

setTimeout(() => {
  $(".progressive-disclosure-container").one("mousemove", function () {
    $(this).find(".progressive-disclosure").removeClass("d-none");
  });
}, 2000);
