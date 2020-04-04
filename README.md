# On food, bias and seasons: <br> A recipe for sustainability
Github repository for `Recipe2BetterRecipe`, an approach for replacing ingredients in recipes that takes into account location and seasonality in order to lower greenhouse gas emissions. This is done by modeling recipes using an RBM and then sampling ingredients using the RBM distributions constrained to use only seasonal ingredients.

## Usage

Download recipes from the Kaggle [What's cooking?](https://www.kaggle.com/c/whats-cooking-kernels-only) under recipes/train.json and recipes/test.json. <br>
Please read carefully the provided [rules](https://www.kaggle.com/c/whats-cooking-kernels-only/rules).

Run `Recipe2BetterRecipe.ipynb` in jupyter with python 3.5+

## Credits

**Food contribution** to **GHG emissions** from [Bon Pour le Climat](https://www.bonpourleclimat.org/).

**Recipes** from the Kaggle [What's cooking?](https://www.kaggle.com/c/whats-cooking-kernels-only)

**Seasonality table** from [Yves Leers and Jean-Luc Fessard](http://www.buchetchastel.fr/ca-chauffe-dans-nos-assiettes-yves-leers-9782283030219).

## Paper

English: [On food, bias and seasons: A recipe for sustainability](paper/VO.pdf).

French: [Sur notre alimentation, nos biais et nos saisons: Vers des recettes durables et responsables](paper/VF.pdf)

## Project contribution

Check [local-seasonal.org/about](https://www.local-seasonal.org/about)

Below is a short **todo list** (suggestions) to contribute to this repository. <br>
Feel free to suggest other ideas and contribute to the project:

* Quantitatively measure how well the new ingredients fit with the other ingredients in a recipe, <br> e.g. with Human judgement or a discriminator.

* Improve code readability and reusability. Add documentation and tests.

* Edit `seasons/paris-fruits.txt` and `seasons/paris-vegs.txt` with what's truely local-seasonal in Paris (produced within 250km).

* Add more locations and seasonality tables. <br> Why not start with your city? :)

## License
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
