# export_variables.py
import pandas as pd
import json

RATING_VARIABLES = {
    # High correlation with carbon intensity, clearly observable in descriptions
    "process_emission_intensity": {
        "description": "Likelihood of inherent process emissions from chemical reactions or high-temperature processing",
        "scale_1": "Low - assembly, fabrication, services, simple mechanical processing",
        "scale_1_examples": "Legal services (541100), Software publishers (511200)",
        "scale_4": "Moderate - some chemical processing, heat treatment, smelting",
        "scale_4_examples": "Paper mills (322120), Glass product manufacturing (327200)",
        "scale_7": "High - primary production of cement, steel, chemicals, refining",
        "scale_7_examples": "Cement manufacturing (327310), Iron and steel mills (331110)",
        "category": "Process"
    },
    
    "material_processing_depth": {
        "description": "How extensively raw materials are transformed",
        "scale_1": "Primary transformation - raw material to basic form (mining, milling, smelting)",
        "scale_1_examples": "Coal mining (212100), Stone mining and quarrying (212310)",
        "scale_4": "Secondary processing - shaping, forming, chemical processing",
        "scale_4_examples": "Millwork (321910), Plastics product manufacturing (326190)",
        "scale_7": "Final assembly/finishing - component assembly, packaging, quality control",
        "scale_7_examples": "Automobile manufacturing (336111), Electronic computer manufacturing (334111)",
        "category": "Process"
    },
    
    "thermal_process_intensity": {
        "description": "Presence of high-temperature processes requiring significant heat",
        "scale_1": "Ambient/low-heat operations - assembly, cold processing, services",
        "scale_1_examples": "Legal services (541100), Software publishers (511200)",
        "scale_4": "Moderate heating - drying, curing, moderate temperature processing",
        "scale_4_examples": "Sawmills and wood preservation (321100), Bread and bakery product manufacturing (311810)",
        "scale_7": "High-heat processes - smelting, refining, cement kilns, glass melting",
        "scale_7_examples": "Cement manufacturing (327310), Glass and glass product manufacturing (327200)",
        "category": "Energy"
    },
    
    "electrification_feasibility": {
        "description": "How easily processes could potentially run on electricity",
        "scale_1": "Already highly electrified - motors, pumps, conveyors, electronics",
        "scale_1_examples": "Data processing and hosting (518200), Electronic computer manufacturing (334111)",
        "scale_4": "Mixed - some processes electrifiable, others require combustion",
        "scale_4_examples": "Paper mills (322120), Food processing (311410)",
        "scale_7": "Hard-to-electrify - high-temperature heat, chemical reduction processes",
        "scale_7_examples": "Cement manufacturing (327310), Iron and steel mills (331110)",
        "category": "Energy"
    },
    
    "continuous_operations_intensity": {
        "description": "Degree to which operations run continuously vs. batch/intermittent",
        "scale_1": "Primarily continuous - 24/7 operations, baseload energy demand",
        "scale_1_examples": "Electric power generation (221100), Petroleum refineries (324110)",
        "scale_4": "Mixed - some continuous, some batch",
        "scale_4_examples": "Chemical manufacturing (325190), Plastics product manufacturing (326190)",
        "scale_7": "Primarily batch/intermittent - start-stop, variable energy demand",
        "scale_7_examples": "Construction machinery manufacturing (333120), Furniture manufacturing (337110)",
        "category": "Operations"
    },
    
    "material_throughput_scale": {
        "description": "Relative volume/scale of material processing",
        "scale_1": "Large-scale/commodity - high volume, mass production",
        "scale_1_examples": "Iron and steel mills (331110), Petroleum refineries (324110)",
        "scale_4": "Medium-scale - moderate volumes, regional operations",
        "scale_4_examples": "Breweries (312120), Dairy product manufacturing (31151A)",
        "scale_7": "Small-scale/specialty - low volume, specialized products",
        "scale_7_examples": "Custom computer programming (541511), Jewelry manufacturing (339910)",
        "category": "Scale"
    },
    
    "chemical_intensity": {
        "description": "Presence of chemical processes or reactions in production",
        "scale_1": "Minimal - mechanical, physical, assembly operations",
        "scale_1_examples": "Automobile assembly (336111), Furniture manufacturing (337110)",
        "scale_4": "Moderate - some chemical treatment, additives, processing",
        "scale_4_examples": "Pulp mills (322110), Photographic equipment manufacturing (333316)",
        "scale_7": "High - chemical synthesis, refining, primary chemical production",
        "scale_7_examples": "Petrochemical manufacturing (325110), Pharmaceutical preparation (325412)",
        "category": "Process"
    },
    
    "capital_vs_labor_intensity": {
        "description": "Balance between capital equipment and manual labor (proxy for automation/modernization)",
        "scale_1": "Capital-intensive - highly automated, equipment-heavy, low labor per unit",
        "scale_1_examples": "Petroleum refineries (324110), Electric power generation (221100)",
        "scale_4": "Balanced - mix of automation and manual operations",
        "scale_4_examples": "Machine shops (332710), Printing (323110)",
        "scale_7": "Labor-intensive - handcrafted, manual processes, high labor input",
        "scale_7_examples": "Jewelry manufacturing (339910), Private households (814000)",
        "category": "Structure"
    }
}

# Save variables definition to CSV
df_variables = pd.DataFrame([
    {
        'variable': var,
        'description': data['description'],
        'scale_1': data['scale_1'],
        'scale_1_examples': data['scale_1_examples'],
        'scale_4': data['scale_4'],
        'scale_4_examples': data['scale_4_examples'],
        'scale_7': data['scale_7'],
        'scale_7_examples': data['scale_7_examples'],
        'category': data['category']
    }
    for var, data in RATING_VARIABLES.items()
])

df_variables.to_csv('carbon_intensity_variables.csv', index=False)
print("✅ Variables exported to carbon_intensity_variables.csv")
print(f"📊 Total variables: {len(RATING_VARIABLES)}")
print("\nCategories:")
for category in set(data['category'] for data in RATING_VARIABLES.values()):
    count = sum(1 for data in RATING_VARIABLES.values() if data['category'] == category)
    print(f"  • {category}: {count} variables")


# # Old variables
# RATING_VARIABLES = {
#     # Energy Source Variables
#     "fossil_fuel_dependency": {
#         "description": "Reliance on coal, oil, and gas for direct energy needs",
#         "scale_1": "Minimal fossil fuel use (<20%, mostly renewables/electricity)",
#         "scale_3": "Mixed fuel sources (40-60% fossil fuels)",
#         "scale_5": "Very high fossil fuel dependency (>80%, coal/oil dominant)",
#         "category": "Energy Source"
#     },
#     "fuel_switching_potential": {
#         "description": "Ability to switch from high-carbon to lower-carbon fuels",
#         "scale_1": "Already using optimal low-carbon fuels",
#         "scale_3": "Some processes could switch with investment",
#         "scale_5": "Locked into high-carbon fuels (no alternatives)",
#         "category": "Energy Source"
#     },
#     "renewable_penetration": {
#         "description": "Current use of renewables or low-carbon feedstocks",
#         "scale_1": "High renewable use (>50%, solar/wind/biomass)",
#         "scale_3": "Moderate renewable use (20-50%)",
#         "scale_5": "Low or no renewable use (<10%)",
#         "category": "Energy Source"
#     },
    
#     # Grid/Indirect Emissions
#     "grid_decarbonization_dependency": {
#         "description": "Reliance on grid electricity and its likely carbon intensity",
#         "scale_1": "Uses renewable PPAs or own zero-carbon generation",
#         "scale_3": "Mixed grid with some renewable share",
#         "scale_5": "Heavy reliance on coal-intensive grid",
#         "category": "Indirect Emissions"
#     },
#     "electrification_readiness": {
#         "description": "Potential to electrify processes currently using direct combustion",
#         "scale_1": "Already highly electrified",
#         "scale_3": "Some processes can electrify with technology change",
#         "scale_5": "Hard-to-electrify processes dominate (high-temp heat)",
#         "category": "Indirect Emissions"
#     },
    
#     # Process Emissions
#     "process_emission_intensity": {
#         "description": "Inherent process emissions from chemical reactions",
#         "scale_1": "Low process emissions (assembly, fabrication, services)",
#         "scale_3": "Moderate process emissions (some chemical processing)",
#         "scale_5": "High process emissions (cement, steel, chemicals primary production)",
#         "category": "Process"
#     },
#     "ccus_deployment": {
#         "description": "Current use of carbon capture technology",
#         "scale_1": "CCUS deployed or in advanced planning",
#         "scale_3": "Piloting or studying CCUS options",
#         "scale_5": "No CCUS plans or considerations",
#         "category": "Process"
#     },
    
#     # Material Efficiency
#     "material_efficiency_potential": {
#         "description": "Opportunity for lightweighting and efficient material use",
#         "scale_1": "Already optimized material use (industry best practice)",
#         "scale_3": "Moderate improvement potential identified",
#         "scale_5": "Significant waste, overdesign, or inefficiency",
#         "category": "Material"
#     },
#     "circular_economy_adoption": {
#         "description": "Use of recycled materials and design for recyclability",
#         "scale_1": "High recycled content (>50%, designed for circularity)",
#         "scale_3": "Moderate recycled content (20-50%)",
#         "scale_5": "Virgin materials only, no design for recycling",
#         "category": "Material"
#     },
#     "product_lifetime_strategy": {
#         "description": "Design approach to product longevity",
#         "scale_1": "Designed for long life, repair, and upgrade",
#         "scale_3": "Standard industry lifetime with some durability",
#         "scale_5": "Planned obsolescence or disposable design",
#         "category": "Material"
#     }
# }