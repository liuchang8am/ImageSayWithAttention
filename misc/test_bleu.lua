require './bleu_scorer'

A = BleuScorer{single_sample=true}

generated = {} -- generated sentences
table.insert(generated,"a dog is running on the dog grass ")
--table.insert(generated,"a group of people are playing basketball ")

ground_truth = {} -- ground truth sentences
table.insert(ground_truth,"a cat is running on the cat sea")
--table.insert(ground_truth,"two men are playing basketball on a court")

function compute_score(ground_truth, generated)
    for k,v in pairs(ground_truth) do
	hypo = generated[k]
	ref = ground_truth[k]
	A:_add(hypo, ref)
    end
    scores = A:compute_score()
    return scores
end

print (compute_score(ground_truth, generated))
