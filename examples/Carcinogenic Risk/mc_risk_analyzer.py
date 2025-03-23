import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from src.core import MonteCarloSimulation
from src.samplers.stratified_sampler import StratifiedSampler

class CarcinogenicRiskAnalyzer:
    def __init__(self, data_file):
        """Initialize the analyzer with data file."""
        self.data_file = data_file
        self.n_simulations = 10000  # As used in the paper
        self.acceptable_range = (1e-6, 1e-4)  # Carcinogenic risk acceptable range
        self.load_data()

    def load_data(self):
        """Load and process the data file."""
        try:
            if self.data_file.endswith('.csv'):
                self.data = pd.read_csv(self.data_file)
            else:
                raise ValueError("Unsupported file format. Please use CSV file.")
            
            # Extract columns based on fixed indices
            self.cd_ingestion = self.data.iloc[:, 1]  # Cd column for ingestion
            self.pb_ingestion = self.data.iloc[:, 2]  # Pb column for ingestion
            self.total_ingestion = self.data.iloc[:, 3]  # Total ingestion risk
            self.cd_dermal = self.data.iloc[:, 4]     # Cd column for dermal
            self.pb_dermal = self.data.iloc[:, 5]     # Pb column for dermal
            self.total_dermal = self.data.iloc[:, 6]  # Total dermal risk
            
            # Convert to numeric, replacing any non-numeric values with NaN
            self.cd_ingestion = pd.to_numeric(self.cd_ingestion, errors='coerce')
            self.pb_ingestion = pd.to_numeric(self.pb_ingestion, errors='coerce')
            self.total_ingestion = pd.to_numeric(self.total_ingestion, errors='coerce')
            self.cd_dermal = pd.to_numeric(self.cd_dermal, errors='coerce')
            self.pb_dermal = pd.to_numeric(self.pb_dermal, errors='coerce')
            self.total_dermal = pd.to_numeric(self.total_dermal, errors='coerce')
            
            # Calculate baseline statistics
            print("\nBaseline Risk Statistics:")
            total_risk = self.cd_ingestion + self.pb_ingestion + self.cd_dermal + self.pb_dermal
            print(f"Mean Risk: {total_risk.mean():.2e}")
            print(f"Risk Range: {total_risk.min():.2e} to {total_risk.max():.2e}")
            
            # Validate against provided total risks
            provided_total = self.total_ingestion + self.total_dermal
            relative_diff = abs(total_risk - provided_total) / provided_total
            print("\nValidation against provided totals:")
            print(f"Max relative difference: {relative_diff.max():.2%}")
            print(f"Mean relative difference: {relative_diff.mean():.2%}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def calculate_risk_probabilities(self, total_risks):
        """Calculate probabilities of risk falling in different ranges."""
        total_risks = total_risks.numpy() if isinstance(total_risks, torch.Tensor) else total_risks
        
        # Calculate probabilities as percentages
        prob_exceed = np.mean(total_risks > self.acceptable_range[1]) * 100
        prob_below = np.mean(total_risks < self.acceptable_range[0]) * 100
        prob_acceptable = 100 - prob_exceed - prob_below
        
        results = {
            'Below Acceptable (<1e-6)': prob_below,
            'Within Acceptable Range': prob_acceptable,
            'Exceeds Acceptable (>1e-4)': prob_exceed
        }
        
        # Print detailed probability analysis
        print("\nRisk Probability Analysis:")
        print(f"Probability of Exceeding Acceptable Risk (>1e-4): {prob_exceed:.2f}%")
        print(f"Probability Within Acceptable Range: {prob_acceptable:.2f}%")
        print(f"Probability Below Minimum Risk (<1e-6): {prob_below:.2f}%")
        
        return results

    def plot_risk_distribution(self, total_risks):
        """Plot the distribution of risk values with detailed analysis."""
        total_risks = total_risks.numpy() if isinstance(total_risks, torch.Tensor) else total_risks
        
        plt.figure(figsize=(12, 8))
        
        # Create histogram with logarithmic bins
        bins = np.logspace(np.log10(total_risks.min()), np.log10(total_risks.max()), 50)
        
        # Plot histogram for different risk ranges
        plt.hist(total_risks[total_risks < self.acceptable_range[0]], 
                bins=bins, alpha=0.7, color='green', label='Below acceptable range (<1e-6)')
        plt.hist(total_risks[(total_risks >= self.acceptable_range[0]) & 
                           (total_risks <= self.acceptable_range[1])], 
                bins=bins, alpha=0.7, color='blue', label='Within acceptable range')
        plt.hist(total_risks[total_risks > self.acceptable_range[1]], 
                bins=bins, alpha=0.7, color='red', label='Above acceptable range (>1e-4)')
        
        # Add vertical lines for acceptable range
        plt.axvline(x=self.acceptable_range[0], color='k', linestyle='--', label='Lower Limit (1e-6)')
        plt.axvline(x=self.acceptable_range[1], color='k', linestyle='--', label='Upper Limit (1e-4)')
        
        # Calculate and add statistics
        percentile_5 = np.percentile(total_risks, 5)
        percentile_95 = np.percentile(total_risks, 95)
        
        stats_text = f'Risk Statistics:\n'
        stats_text += f'Mean: {np.mean(total_risks):.2e}\n'
        stats_text += f'Median: {np.median(total_risks):.2e}\n'
        stats_text += f'5th percentile: {percentile_5:.2e}\n'
        stats_text += f'95th percentile: {percentile_95:.2e}\n'
        stats_text += f'\nRisk Range:\n'
        stats_text += f'Min: {np.min(total_risks):.2e}\n'
        stats_text += f'Max: {np.max(total_risks):.2e}'
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.title('Distribution of Carcinogenic Risk Values\n(10,000 Monte Carlo Simulations)')
        plt.xlabel('Risk Value')
        plt.ylabel('Frequency')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save plot
        plt.savefig('risk_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_metal_contribution(self):
        """Analyze and visualize metal contributions using Monte Carlo results."""
        try:
            if not hasattr(self, 'monte_carlo_results'):
                raise ValueError("Monte Carlo simulation must be run before analyzing contributions")
            
            results = self.monte_carlo_results['contributions']
            
            # Create figure with two subplots with increased height and top margin
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            # Plot 1: Total metal contributions
            data_total = [results['cd_total'], results['pb_total']]
            ax1.boxplot(data_total, tick_labels=['Cd', 'Pb'])
            ax1.set_title('Total Metal Contributions\nto Carcinogenic Risk', pad=35)
            ax1.set_ylabel('Contribution (%)')
            ax1.grid(True, alpha=0.3)
            
            # Add mean values for total contributions with adjusted position
            for i, data in enumerate(data_total, 1):
                mean_val = torch.mean(data)
                std_val = torch.std(data)
                ax1.text(i, 100, f'Mean: {mean_val:.1f}%\nStd: {std_val:.1f}%', 
                        horizontalalignment='center', verticalalignment='bottom')
            
            # Plot 2: Pathway-specific contributions
            data_pathway = [
                results['cd_ing'],
                results['cd_derm'],
                results['pb_ing'],
                results['pb_derm']
            ]
            labels = ['Cd-Ingestion', 'Cd-Dermal', 'Pb-Ingestion', 'Pb-Dermal']
            ax2.boxplot(data_pathway, tick_labels=labels)
            ax2.set_title('Exposure Pathway\nContributions', pad=35)
            ax2.set_ylabel('Contribution (%)')
            ax2.grid(True, alpha=0.3)
            
            # Add mean and std values for pathway contributions with adjusted position
            for i, data in enumerate(data_pathway, 1):
                mean_val = torch.mean(data)
                std_val = torch.std(data)
                ax2.text(i, ax2.get_ylim()[1], f'Mean: {mean_val:.1f}%\nStd: {std_val:.1f}%', 
                        horizontalalignment='center', verticalalignment='bottom')
            
            # Use figure handle directly for title and layout
            fig.suptitle('Metal and Pathway Contributions to Carcinogenic Risk\n(Monte Carlo Analysis)', y=1.1)
            fig.tight_layout()
            
            # Save with extra space for the title
            fig.savefig('metal_contributions.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
            plt.close(fig)
            
            # Print detailed analysis
            print("\nMetal Contribution Analysis:")
            cd_mean = torch.mean(results['cd_total']).item()
            pb_mean = torch.mean(results['pb_total']).item()
            print(f"\nTotal Metal Contributions:")
            print(f"Cadmium (Cd): {cd_mean:.1f}% ± {torch.std(results['cd_total']):.1f}%")
            print(f"Lead (Pb): {pb_mean:.1f}% ± {torch.std(results['pb_total']):.1f}%")
            print(f"\nLeading Cause: {'Cadmium (Cd)' if cd_mean > pb_mean else 'Lead (Pb)'}")
            
            print("\nExposure Pathway Contributions:")
            print(f"Cd Ingestion: {torch.mean(results['cd_ing']):.1f}% ± {torch.std(results['cd_ing']):.1f}%")
            print(f"Cd Dermal: {torch.mean(results['cd_derm']):.1f}% ± {torch.std(results['cd_derm']):.1f}%")
            print(f"Pb Ingestion: {torch.mean(results['pb_ing']):.1f}% ± {torch.std(results['pb_ing']):.1f}%")
            print(f"Pb Dermal: {torch.mean(results['pb_derm']):.1f}% ± {torch.std(results['pb_derm']):.1f}%")
            
            return {
                'Metals': {
                    'Cd': (float(torch.mean(results['cd_total'])), float(torch.std(results['cd_total']))),
                    'Pb': (float(torch.mean(results['pb_total'])), float(torch.std(results['pb_total'])))
                },
                'Pathways': {
                    'Cd_ingestion': (float(torch.mean(results['cd_ing'])), float(torch.std(results['cd_ing']))),
                    'Cd_dermal': (float(torch.mean(results['cd_derm'])), float(torch.std(results['cd_derm']))),
                    'Pb_ingestion': (float(torch.mean(results['pb_ing'])), float(torch.std(results['pb_ing']))),
                    'Pb_dermal': (float(torch.mean(results['pb_derm'])), float(torch.std(results['pb_derm'])))
                }
            }
            
        except Exception as e:
            print(f"Error in metal contribution analysis: {e}")
            raise

    def monte_carlo_simulation(self):
        """Run Monte Carlo simulation using proper simulation infrastructure."""
        try:
            # Setup dimensions for sampling (4 dimensions: Cd-ing, Pb-ing, Cd-derm, Pb-derm)
            dimension = 4
            
            # Initialize sampler with stratification
            sampler = StratifiedSampler(
                dimension=dimension,
                strata_per_dim=6,  # 6 strata per dimension for better coverage
                device="cpu"
            )
            
            # Store simulation results
            all_contributions = []
            all_risks = []
            
            # Convert data to numpy arrays for easier handling
            cd_ing_data = self.cd_ingestion.values
            pb_ing_data = self.pb_ingestion.values
            cd_derm_data = self.cd_dermal.values
            pb_derm_data = self.pb_dermal.values
            
            # Calculate and print initial statistics
            print("\nOriginal Data Statistics:")
            for name, data in [
                ("Cd Ingestion", cd_ing_data),
                ("Pb Ingestion", pb_ing_data),
                ("Cd Dermal", cd_derm_data),
                ("Pb Dermal", pb_derm_data)
            ]:
                print(f"{name}: Mean = {np.mean(data):.2e}, Std = {np.std(data):.2e}")
            
            # Define target function for simulation
            def target_function(samples):
                batch_size = samples.shape[0]
                
                # Randomly select actual data points as base values
                indices = torch.randint(0, len(cd_ing_data), (batch_size,))
                
                # Get base values from actual data
                cd_ing_base = torch.tensor([cd_ing_data[i] for i in indices])
                pb_ing_base = torch.tensor([pb_ing_data[i] for i in indices])
                cd_derm_base = torch.tensor([cd_derm_data[i] for i in indices])
                pb_derm_base = torch.tensor([pb_derm_data[i] for i in indices])
                
                # Apply variations using samples
                variation_scale = 0.2  # 20% variation
                cd_ing = cd_ing_base * torch.exp(variation_scale * samples[:, 0])
                pb_ing = pb_ing_base * torch.exp(variation_scale * samples[:, 1])
                cd_derm = cd_derm_base * torch.exp(variation_scale * samples[:, 2])
                pb_derm = pb_derm_base * torch.exp(variation_scale * samples[:, 3])
                
                # Calculate total risk and contributions
                total_risk = cd_ing + pb_ing + cd_derm + pb_derm
                
                # Store total risks for later analysis
                all_risks.append(total_risk.detach())
                
                # Calculate contribution percentages
                cd_total = (cd_ing + cd_derm) / total_risk * 100
                pb_total = (pb_ing + pb_derm) / total_risk * 100
                cd_ing_contrib = cd_ing / total_risk * 100
                cd_derm_contrib = cd_derm / total_risk * 100
                pb_ing_contrib = pb_ing / total_risk * 100
                pb_derm_contrib = pb_derm / total_risk * 100
                
                # Store all contributions for later use
                contributions = torch.stack([cd_total, pb_total, cd_ing_contrib, cd_derm_contrib, pb_ing_contrib, pb_derm_contrib], dim=1)
                all_contributions.append(contributions.detach())
                
                return total_risk
            
            # Create and run simulation
            simulation = MonteCarloSimulation(
                sampler=sampler,
                target_function=target_function,
                n_samples=self.n_simulations,
                batch_size=1000,
                use_gpu=torch.cuda.is_available(),
                seed=42
            )
            
            print("\nRunning Monte Carlo simulation with 10,000 iterations...")
            simulation.run()  # We don't need to store the result as we're using all_risks directly
            
            # Process contribution results
            all_contribution_tensors = torch.cat(all_contributions)
            all_risks_tensor = torch.cat(all_risks)
            
            # Store results for plotting
            self.monte_carlo_results = {
                'total_risks': all_risks_tensor,
                'contributions': {
                    'cd_total': all_contribution_tensors[:, 0],
                    'pb_total': all_contribution_tensors[:, 1],
                    'cd_ing': all_contribution_tensors[:, 2],
                    'cd_derm': all_contribution_tensors[:, 3],
                    'pb_ing': all_contribution_tensors[:, 4],
                    'pb_derm': all_contribution_tensors[:, 5]
                }
            }
            
            # Calculate and print risk probabilities
            self.calculate_risk_probabilities(all_risks_tensor)
            
            # Plot risk distribution
            self.plot_risk_distribution(all_risks_tensor)
            
        except Exception as e:
            print(f"Error in Monte Carlo simulation: {e}")
            raise

def main():
    # Initialize analyzer with data file
    data_file = 'sample.csv'
    analyzer = CarcinogenicRiskAnalyzer(data_file)
    
    # Run Monte Carlo simulation
    analyzer.monte_carlo_simulation()
    
    # Analyze metal contributions
    analyzer.analyze_metal_contribution()
    
    print("\nAnalysis complete. Results have been saved to:")
    print("1. 'risk_distribution.png' - Shows the distribution of carcinogenic risk values")
    print("2. 'metal_contributions.png' - Shows metal and pathway contributions")

if __name__ == "__main__":
    main()