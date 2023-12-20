# comment analysis experiment

Enter comments -> Fix typos -> Get sentiments, semantic similarity graph, and comment groups/classes.

Live demo: https://comment-analysis.surge.sh (`yarn deploy` after running `yarn dev`)

Local demo: `yarn dev` (if you run into `Error: Expected content key ### to exist`, try running `rm -rf .parcel-cache` first)

## references and things this projects builds on

[learning-tfjs-umap](https://github.com/hchiam/learning-tfjs-umap)

[tfjs](https://github.com/tensorflow/tfjs)

[USE (Universal Sentence Encoder)](https://github.com/tensorflow/tfjs-models/blob/master/universal-sentence-encoder/README.md) (see my [text-similarity-test](https://github.com/hchiam/text-similarity-test))

[KNN](https://github.com/tensorflow/tfjs-models/blob/master/knn-classifier/README.md)

[umap-js](https://github.com/PAIR-code/umap-js#umap-js) ([UMAP vs t-SNE vs SNE](https://towardsdatascience.com/visualizing-your-embeddings-4c79332581a9))

[nlp.js](https://github.com/axa-group/nlp.js) (note: `SpellCheck` currently seems incompatible with one of [parcel](https://github.com/hchiam/learning-parcel#learning-parceljs)'s transformers)

[natural](https://github.com/NaturalNode/natural)

[dictionaries](https://github.com/wooorm/dictionaries#example-use-with-nspell)

[nspell](https://github.com/wooorm/nspell)

[jQuery](https://github.com/hchiam/learning-jquery#learning-jquery)

[chart.js](https://github.com/chartjs/Chart.js)

(otherwise just look at [package.json](https://github.com/hchiam/comment-analysis/blob/main/package.json))

## typo fix suggestions under input box to fix input box

- https://github.com/axa-group/nlp.js/blob/master/docs/v4/similarity.md#spellcheck (seems to have compatibility issue with parcel)
- https://github.com/axa-group/nlp.js/blob/master/docs/v4/similarity.md#spellcheck-trained-with-words-trained-from-a-text
- https://naturalnode.github.io/natural/spellcheck.html (doesn't come with a dictionary)
- https://github.com/wooorm/dictionaries#example-use-with-nspell (provides a dictionary like `dictionary-en` or `dictionary-en-ca` or [more languages](https://github.com/wooorm/dictionaries/tree/main/dictionaries), but you'll likely have to [manually copy and include some specific code](https://github.com/wooorm/dictionaries/issues/51#issuecomment-1627801903))

## sentiments above chart

- https://github.com/axa-group/nlp.js/blob/master/docs/v3/sentiment-analysis.md
- https://naturalnode.github.io/natural/sentiment_analysis.html

## knn under chart

- KNN for automatic grouping, maybe with tweakable k input box: https://github.com/tensorflow/tfjs-models/blob/master/knn-classifier/README.md
