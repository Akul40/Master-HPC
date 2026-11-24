import os
import json
import re
import time
import boto3

from botocore.exceptions import ClientError, BotoCoreError


# --------------------------------------------------
# 1. AMAZON BEDROCK CLIENT
# --------------------------------------------------

def get_bedrock_client():

    region = (
        os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "us-east-1"
    )

    return boto3.client(
        service_name="bedrock-runtime",
        region_name=region
    )


def get_model():

    model_id = os.getenv("BEDROCK_MODEL_ID")

    if not model_id:
        raise ValueError(
            "Bedrock model is not configured. "
            "Set BEDROCK_MODEL_ID before starting the backend."
        )

    return model_id


def get_fallback_models():
    return[]


# --------------------------------------------------
# 2. BEDROCK GENERATION WITH RETRIES AND FALLBACK
# --------------------------------------------------

def generate_with_fallback(client, prompt, config=None):

    primary_model = get_model()

    # Remove duplicate model IDs
    models = list(dict.fromkeys(
        [primary_model] + get_fallback_models()
    ))

    errors = []

    retryable_errors = [
        "ThrottlingException",
        "ServiceUnavailableException",
        "InternalServerException",
        "ModelTimeoutException",
        "ModelNotReadyException"
    ]

    for model in models:

        print(f"Trying Amazon Bedrock model: {model}")

        for attempt in range(2):

            try:

                response = client.converse(
                    modelId=model,

                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "text": prompt
                                }
                            ]
                        }
                    ],

                    inferenceConfig={
                        "maxTokens": 4000,
                        "temperature": 0.2
                    }
                )

                # Extract generated text
                content = (
                    response
                    .get("output", {})
                    .get("message", {})
                    .get("content", [])
                )

                response_text = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict)
                ).strip()

                if response_text:

                    print(
                        f"Bedrock response received from {model}"
                    )

                    return response_text

                raise ValueError(
                    "Amazon Bedrock returned an empty response."
                )

            except ClientError as e:

                error_code = (
                    e.response
                    .get("Error", {})
                    .get("Code", "UnknownError")
                )

                error_message = str(e)

                errors.append(
                    f"{model}: {error_code} - {error_message}"
                )

                if error_code in retryable_errors:

                    if attempt == 0:

                        print(
                            f"{model} temporarily unavailable. "
                            "Retrying..."
                        )

                        time.sleep(2)
                        continue

                    print(
                        f"{model} failed after retry. "
                        "Moving to the next model."
                    )

                    break

                print(
                    f"Bedrock model {model} failed: {error_code}. "
                    "Moving to the next model."
                )

                break

            except (BotoCoreError, ValueError) as e:

                errors.append(
                    f"{model}: {str(e)}"
                )

                print(
                    f"Error with Bedrock model {model}: {e}"
                )

                break

            except Exception as e:

                errors.append(
                    f"{model}: {str(e)}"
                )

                print(
                    f"Unexpected error with {model}: {e}"
                )

                break

    raise RuntimeError(
        "All Amazon Bedrock models failed:\n"
        + "\n".join(errors)
    )


# --------------------------------------------------
# 3. CLUSTERING ANALYSIS
# --------------------------------------------------

def analyze_clustering(results):

    client = get_bedrock_client()

    actual_results = json.dumps(
        results,
        indent=2,
        default=str
    )

    prompt = f"""
You are a Healthcare Commercial Analytics Consultant.

Your audience consists of:

- Brand Managers
- Sales Managers
- Commercial Excellence Teams
- Marketing Teams
- Business Leaders

The audience does NOT have data science expertise.

Your job is to explain segmentation results in clear business language.

IMPORTANT:

Base all conclusions ONLY on the supplied results.

Do not invent findings.

Do not assume business opportunities, future potential,
segment value, or commercial impact unless directly
supported by the supplied data.

If information is not available, state:
"Additional data is required."

==================================================
CLUSTERING RESULTS
==================================================

{actual_results}

==================================================
OUTPUT FORMAT
==================================================

# Executive Summary

Provide:

- Number of HCPs analysed
- Number of methods evaluated
- Best performing method
- Suggested segment count
- Top 3 key observations

Maximum 120 words.

Avoid technical terminology.

==================================================
# Segment Snapshot
==================================================

Provide a simple table:

| Segment | HCP Count | Prescription Activity | Engagement Level |

Create business-friendly segment names when supported
by the data.

Examples:

- High Activity HCPs
- Moderate Activity HCPs
- Lower Activity HCPs

Do not invent labels that are unsupported by the data.

==================================================
# Segment Profiles
==================================================

For each segment in the best-performing method:

Provide:

### Segment Name

- Number of HCPs
- Average prescription behaviour
- Average engagement behaviour
- Important categorical characteristics
- Key differentiators

Business Interpretation:

Write a short paragraph describing the observed behaviour.

Only describe facts visible in the supplied data.

Do not infer commercial opportunity.

Do not infer future behaviour.

Do not infer HCP value.

If evidence is insufficient, say so.

==================================================
# Model Comparison
==================================================

Create a table:

| Method | Segments | Silhouette Score | Summary |

After the table provide:

- Best performing method
- Reason for selection

Maximum 100 words.

Important:

Do not recommend a model using silhouette score alone.

Consider:

- Segment stability
- Coverage
- Interpretability

If K-Means and Hierarchical create the same partition,
mention this.

==================================================
# Segment Count Assessment
==================================================

Using elbow_analysis:

State:

- Highest silhouette score
- Elbow recommendation
- Whether both metrics agree

Maximum 100 words.

Use simple language.

Do not explain mathematical formulas.

Do not explain inertia calculations.

If silhouette and elbow differ, clearly state that
both options may be valid.

==================================================
# Why These Segments Were Created
==================================================

Explain how clustering was performed.

Use business language.

Explain that:

- Similar HCPs are grouped together.
- Selected features were used.
- Higher feature weights had greater influence.

If feature weights are supplied, explain their impact.

Maximum 120 words.

==================================================
# Outlier Review
==================================================

Only if DBSCAN identified noise points:

State:

- Number of outliers
- Exact HCP IDs from the supplied data
- Exact HCP names from the supplied data

Do not modify names.

Do not abbreviate names.

Do not invent names.

Explain that outliers simply behave differently from
the main groups.

Do not describe outliers as low value.

If no outliers exist, state that clearly.

==================================================
# Business Recommendations
==================================================

Provide exactly 5 recommendations.

Recommendations must:

- Be practical
- Be business-focused
- Be supported by the supplied results

Examples:

- Validate segments with commercial teams.
- Review behavioural differences across segments.
- Track segment changes over time.

Do not recommend sales actions, marketing actions,
or engagement strategies unless supported by data.

Each recommendation must be one sentence.

==================================================
# Confidence Assessment
==================================================

Provide:

Overall Confidence:
High / Medium / Low

Reason:

Base confidence on:

- Data quality
- Agreement between methods
- Stability of results
- Availability of meaningful separation

Maximum 80 words.

==================================================
IMPORTANT RULES
==================================================

Never invent:

- HCP IDs
- HCP names
- Metrics
- Segment sizes
- Findings
- Business conclusions

Never claim:

- A segment has more opportunity
- A segment has higher future value
- A segment should receive more investment

unless explicitly supported by supplied data.

Do not explain algorithms in technical detail.

Do not explain cluster mathematics.

Do not explain DBSCAN parameters.

Use concise business language.

Prefer bullet points.

Keep the entire report under 900 words.

Only reference information present in the supplied results.
"""

    response_text = generate_with_fallback(
        client=client,
        prompt=prompt
    )

    return response_text


# --------------------------------------------------
# 4. JSON RESPONSE PARSER
# --------------------------------------------------

def parse_json_response(response_text):

    # Remove Markdown code fences if present
    cleaned = re.sub(
        r"```(?:json)?|```",
        "",
        response_text,
        flags=re.IGNORECASE
    ).strip()

    try:

        return json.loads(cleaned)

    except json.JSONDecodeError:

        # Try extracting the JSON object
        match = re.search(
            r"\{.*\}",
            cleaned,
            flags=re.DOTALL
        )

        if match:

            try:
                return json.loads(match.group(0))

            except json.JSONDecodeError:
                pass

    raise ValueError(
        "Amazon Bedrock returned an invalid JSON response."
    )


# --------------------------------------------------
# 5. NATURAL LANGUAGE FEATURE WEIGHT SUGGESTIONS
# --------------------------------------------------

def suggest_feature_adjustments(
    instruction,
    selected_features,
    current_weights
):

    client = get_bedrock_client()

    available_features = list(selected_features)

    # Validate current weights
    weights = {}

    for feature in available_features:

        try:

            value = float(
                current_weights.get(feature, 1)
            )

            if (
                not 0 <= value <= 5
                or not (
                    value == value
                    and abs(value) != float("inf")
                )
            ):

                value = 1

        except (ValueError, TypeError):

            value = 1

        weights[feature] = value

    prompt = f"""
You are an AI assistant helping a business user
configure HCP segmentation.

The user wants to modify the importance of selected
features used by a clustering algorithm.

USER INSTRUCTION:

{instruction}


AVAILABLE FEATURES:

{json.dumps(available_features)}


CURRENT FEATURE WEIGHTS:

{json.dumps(weights)}


YOUR TASK:

Translate the user's instruction into proposed feature weights.

RULES:

1. Only modify features from AVAILABLE FEATURES.

2. Preserve the current weight of features that the
   instruction does not mention.

3. Use numerical weights between 0 and 5.

4. A weight of:
   - 0 means exclude the feature.
   - 1 means normal importance.
   - Above 1 means increased importance.
   - Between 0 and 1 means reduced importance.

5. Do not invent new features.

6. Do not change clustering algorithms or cluster counts.

7. If the instruction is unclear, preserve all current
   weights and explain what needs clarification.

8. Do not claim that a proposed weight will guarantee
   a particular clustering outcome.

9. Return every available feature in proposed_weights.

Return ONLY valid JSON in this exact structure:

{{
    "proposed_weights": {{
        "feature_name": 1.0
    }},
    "explanation": "Explain the proposed changes in simple business language."
}}
"""

    response_text = generate_with_fallback(
        client=client,
        prompt=prompt
    )

    # Parse Bedrock JSON response
    suggestion = parse_json_response(response_text)

    proposed_weights = suggestion.get(
        "proposed_weights",
        {}
    )

    if not isinstance(proposed_weights, dict):
        proposed_weights = {}

    # Validate proposed weights
    validated_weights = {}

    for feature in available_features:

        proposed_value = proposed_weights.get(
            feature,
            weights[feature]
        )

        try:

            proposed_value = float(proposed_value)

        except (ValueError, TypeError):

            proposed_value = weights[feature]

        # Reject NaN, infinity and out-of-range values
        if (
            not 0 <= proposed_value <= 5
            or not (
                proposed_value == proposed_value
                and abs(proposed_value) != float("inf")
            )
        ):

            proposed_value = weights[feature]

        validated_weights[feature] = round(
            proposed_value,
            2
        )

    explanation = suggestion.get(
        "explanation",
        "Feature weights have been proposed based on your instruction."
    )

    if not isinstance(explanation, str):
        explanation = str(explanation)

    return {

        "proposed_weights": validated_weights,

        "explanation": explanation

    }

