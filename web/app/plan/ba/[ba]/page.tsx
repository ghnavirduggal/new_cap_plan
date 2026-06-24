import PlanDetailClient from "../../plan-detail-client";

type PageProps = {
  params: Promise<{ ba: string }>;
};

export default async function BaRollupPage({ params }: PageProps) {
  const { ba } = await params;
  return <PlanDetailClient rollupBa={decodeURIComponent(ba || "")} />;
}
