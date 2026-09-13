from agent import run_agent


question = input(
    "\nAsk something: "
)


response = run_agent(
    question
)


print("\n" + "=" * 60)
print("MODEL RESPONSE")
print("=" * 60)

print(response.content)


print("\n" + "=" * 60)
print("TOOL CALLS")
print("=" * 60)

print(response.tool_calls)