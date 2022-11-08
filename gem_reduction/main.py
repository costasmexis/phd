from cobra.io.json import load_json_model

model = load_json_model('toy_example.json')

# Check model
print(len(model.reactions))
print(len(model.metabolites))
print(len(model.genes))

print(model.objective)