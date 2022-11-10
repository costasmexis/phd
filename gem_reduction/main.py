import pandas as pd
import numpy as np

def single_double_triple_deletions(cobra_model):
        
    # script for single deletion of every reaction in GEM

    df_rxn_del = pd.DataFrame()

    for rxn in cobra_model.reactions:
        df_rxn_del[rxn.id] = np.nan

    rxn_names = []
    growth = []
    status = []

    for rxn in range(len(cobra_model.reactions)):
        with cobra_model:
            cobra_model.reactions[rxn].knock_out()
            solution =  cobra_model.optimize()
            df_rxn_del.loc[rxn] = solution.fluxes
            rxn_names.append(cobra_model.reactions[rxn].id)
            growth.append(solution.objective_value)
            status.append(solution.status)

    df_rxn_del['id'] = rxn_names
    df_rxn_del['growth'] = growth
    df_rxn_del['status'] = status

    # script for double deletions

    df_rxn_double_del = pd.DataFrame()

    for rxn in cobra_model.reactions:
        df_rxn_double_del[rxn.id] = np.nan

    rxn_names = []
    growth = []
    status = []

    counter = 0
    for rxn_a in range(len(cobra_model.reactions)):
        for rxn_b in range(rxn_a, len(cobra_model.reactions)):
            with cobra_model:
                cobra_model.reactions[rxn_a].knock_out()
                cobra_model.reactions[rxn_b].knock_out()
                solution =  cobra_model.optimize()
                rxn_names.append(cobra_model.reactions[rxn_a].id + ',' + cobra_model.reactions[rxn_b].id)
                growth.append(solution.objective_value)
                status.append(solution.status)
                
                df_rxn_double_del.loc[counter] = solution.fluxes
                counter = counter + 1            
                
    df_rxn_double_del['id'] = rxn_names
    df_rxn_double_del['growth'] = growth
    df_rxn_double_del['status'] = status


    # script for triple deletions

    df_rxn_triple_del = pd.DataFrame()

    for rxn in cobra_model.reactions:
        df_rxn_triple_del[rxn.id] = np.nan

    rxn_names = []
    growth = []
    status = []

    counter = 0
    for rxn_a in range(len(cobra_model.reactions)):
        for rxn_b in range(rxn_a, len(cobra_model.reactions)):
            for rxn_c in range(rxn_b, len(cobra_model.reactions)):
                with cobra_model:
                    cobra_model.reactions[rxn_a].knock_out()
                    cobra_model.reactions[rxn_b].knock_out()
                    cobra_model.reactions[rxn_c].knock_out()

                    solution =  cobra_model.optimize()
                    rxn_names.append(cobra_model.reactions[rxn_a].id + ',' + cobra_model.reactions[rxn_b].id + ',' + cobra_model.reactions[rxn_c].id)
                    growth.append(solution.objective_value)
                    status.append(solution.status)
                    
                    df_rxn_triple_del.loc[counter] = solution.fluxes
                    counter = counter + 1            
                
    df_rxn_triple_del['id'] = rxn_names
    df_rxn_triple_del['growth'] = growth
    df_rxn_triple_del['status'] = status


    # Cocnacanate all dfs with deletions
    df = pd.concat([df_rxn_del, df_rxn_double_del, df_rxn_triple_del])
    
    return df







