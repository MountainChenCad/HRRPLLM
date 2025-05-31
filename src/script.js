document.addEventListener('DOMContentLoaded', () => {
    const predictButton = document.getElementById('predict-button');
    const testScsTextarea = document.getElementById('test-scs');
    const llmOutputPre = document.getElementById('llm-output');

    const taskClasses = ['Mirage2000', 'IDF', 'EA-18G'];

    // Prototypes data - Store as actual JavaScript objects/arrays
    const prototypes = [
        {
            name: 'Mirage2000',
            sc_data: [ // Changed from sc_data_str to sc_data (actual objects)
              {'range_index': 467, 'normalized_amplitude': 1.000},
              {'range_index': 462, 'normalized_amplitude': 0.706},
              {'range_index': 563, 'normalized_amplitude': 0.513},
              {'range_index': 573, 'normalized_amplitude': 0.459},
              {'range_index': 475, 'normalized_amplitude': 0.435},
              {'range_index': 484, 'normalized_amplitude': 0.395},
              {'range_index': 439, 'normalized_amplitude': 0.284},
              {'range_index': 450, 'normalized_amplitude': 0.262}
            ]
        },
        {
            name: 'IDF',
            sc_data: [
              {'range_index': 491, 'normalized_amplitude': 1.000}
            ]
        },
        {
            name: 'EA-18G',
            sc_data: [
              {'range_index': 496, 'normalized_amplitude': 1.000},
              {'range_index': 454, 'normalized_amplitude': 0.610},
              {'range_index': 462, 'normalized_amplitude': 0.606},
              {'range_index': 580, 'normalized_amplitude': 0.572},
              {'range_index': 473, 'normalized_amplitude': 0.326},
              {'range_index': 509, 'normalized_amplitude': 0.270},
              {'range_index': 567, 'normalized_amplitude': 0.227}
            ]
        }
    ];

    // Robust parser for SC string from textarea
    function parseSCsFromText(scText) {
        try {
            // Attempt to make it valid JSON-like by replacing single quotes
            // and Pythonic keys if necessary.
            // This is a common source of errors if the input format varies.
            let correctedText = scText.trim();

            // Ensure it looks like an array
            if (!correctedText.startsWith('[') || !correctedText.endsWith(']')) {
                // Try to wrap if it's a single object (though your format is an array)
                if (correctedText.startsWith('{') && correctedText.endsWith('}')) {
                    correctedText = `[${correctedText}]`;
                } else {
                     console.error("Input SCs text does not appear to be an array.");
                    return null;
                }
            }

            // Replace Python-style keys with JS/JSON-style keys
            // Be careful with global replace if keys can be substrings of other text
            correctedText = correctedText.replace(/'range index'/g, '"range_index"');
            correctedText = correctedText.replace(/'normalized amplitude'/g, '"normalized_amplitude"');
            // Replace remaining single quotes around keys/values if they exist
            correctedText = correctedText.replace(/'/g, '"');


            const parsed = JSON.parse(correctedText);

            if (Array.isArray(parsed) && parsed.every(item => typeof item === 'object' && item !== null && 'range_index' in item && 'normalized_amplitude' in item)) {
                return parsed;
            }
            console.error("Parsed data is not a valid array of SC objects:", parsed);
            return null;
        } catch (e) {
            console.error("Error parsing SCs from text:", e, "Input was:", scText, "Corrected attempt:", correctedText);
            return null;
        }
    }


    // Simple heuristic for similarity
    function calculateSimilarityScore(testSCs, prototypeSCs) {
        if (!testSCs || testSCs.length === 0 || !prototypeSCs || prototypeSCs.length === 0) {
            return Infinity;
        }

        let score = 0;
        score += Math.abs(testSCs.length - prototypeSCs.length) * 10;

        const testStrongestPos = testSCs[0].range_index;
        const protoStrongestPos = prototypeSCs[0].range_index;
        score += Math.abs(testStrongestPos - protoStrongestPos);

        return score;
    }

    predictButton.addEventListener('click', () => {
        const testScsStr = testScsTextarea.value;
        const parsedTestSCs = parseSCsFromText(testScsStr); // Use the robust parser

        if (!parsedTestSCs) {
            llmOutputPre.textContent = `Predicted Target Class: Error
Rationale: Could not parse the input Test Sample Scattering Centers. Please ensure it's in the correct format, e.g., an array of objects like: [{'range_index': 490, 'normalized_amplitude': 1.000}]. Check console for details.`;
            return;
        }
        if (parsedTestSCs.length === 0) {
            llmOutputPre.textContent = `Predicted Target Class: Undetermined
Rationale: No scattering centers provided in the test sample.`;
            return;
        }

        let bestMatch = null;
        let lowestScore = Infinity;

        prototypes.forEach(proto => {
            const prototypeSCs = proto.sc_data; // Already parsed objects
            if (prototypeSCs) {
                const score = calculateSimilarityScore(parsedTestSCs, prototypeSCs);
                console.log(`Comparing with ${proto.name}, Score: ${score}`); // Debugging
                if (score < lowestScore) {
                    lowestScore = score;
                    bestMatch = proto;
                }
            }
        });

        if (bestMatch) {
            const bestMatchSCs = bestMatch.sc_data;
            let rationale = `The test sample has ${parsedTestSCs.length} scattering center(s). `;
            rationale += `Its strongest SC is at range index ${parsedTestSCs[0].range_index}. `;
            rationale += `This appears most similar to the '${bestMatch.name}' prototype (similarity score: ${lowestScore.toFixed(2)}), which has ${bestMatchSCs.length} SC(s) with its strongest SC at ${bestMatchSCs[0].range_index}. `;
            if (Math.abs(parsedTestSCs.length - bestMatchSCs.length) <= 1) {
                rationale += `The number of scattering centers is also comparable.`;
            } else {
                rationale += `While the number of scattering centers differs more significantly, the primary SC location and amplitude distribution were considered.`;
            }

            llmOutputPre.textContent = `Predicted Target Class: ${bestMatch.name}\nRationale: ${rationale}`;
        } else {
            llmOutputPre.textContent = `Predicted Target Class: Undetermined
Rationale: Could not find a suitable match among the prototypes based on the provided heuristics.`;
        }
    });
});