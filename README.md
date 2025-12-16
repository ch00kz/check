# Check

> - This is mainly for my own personal use :)
> - Used LLM (claude) to help with type generation and clean-up.

Simple script to track your chess.com progress over time. 

Can aggregate results `daily` or `weekly`

## Examples

### Aggregate weekly for the November to December of 2025
```sh
uv run main.py -u hikaru -m "2025/11-12" -a weekly
```
<img width="1075" height="708" alt="Screenshot 2025-12-11 at 10 31 48 AM" src="https://github.com/user-attachments/assets/ca5ab106-5223-456d-b331-0f2e4cbceea4" />


### Aggregate Daily for the December 2025
```sh
uv run main.py -u hikaru -m "2025/11-12" -a daily
```
<img width="960" height="605" alt="Screenshot 2025-12-11 at 10 32 10 AM" src="https://github.com/user-attachments/assets/c64d4094-8110-438f-aa9d-199a34291621" />


