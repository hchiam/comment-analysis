# Learning tfjs + [umap-js](https://github.com/PAIR-code/umap-js#umap-js)

Just one of the things I'm learning. https://github.com/hchiam/learning

Using Universal-Sentence-Encoder (USE) and UMAP to _attempt_ to graph sentences with semantically-similar meanings close together. UMAP is conceptually similar to PCA in that it also reduces dimensions, but UMAP is stochastic to help with speed of calculation for ML purposes, so the output of UMAP isn't always the same.

Get **UMAP** output coordinates with [umap-js](https://github.com/PAIR-code/umap-js#umap-js) and visualization powered by [chart.js](https://github.com/chartjs/Chart.js).

Great explanation of SNE vs t-SNE (vs UMAP - which sounds better than t-SNE): https://towardsdatascience.com/visualizing-your-embeddings-4c79332581a9 - basically UMAP is better than t-SNE is better than SNE (which is related to the order they were invented). My other repo that uses [tfjs-tsne](https://github.com/tensorflow/tfjs-tsne) can be found here: https://github.com/hchiam/learning-tfjs-tsne

Learn more about TensorFlow at https://github.com/hchiam/learning-tensorflow

## Starting by testing out this repo

Run `yarn global add parcel` or `npm install -g parcel`, and then:

Using [`yarn`](https://github.com/hchiam/learning-yarn):

```bash
git clone https://github.com/hchiam/learning-tfjs-umap.git && cd learning-tfjs-umap && yarn;
yarn dev;
```

Or with `npm`:

```bash
git clone https://github.com/hchiam/learning-tfjs-umap.git && cd learning-tfjs-umap && npm install;
npm run dev;
```

## notes to self

First run `yarn dev` to create the /dist folder, and then run this:

```sh
yarn deploy
```

https://tfjs-umap-demo.surge.sh
